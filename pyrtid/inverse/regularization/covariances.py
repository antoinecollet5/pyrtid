"""Provide covariance matrix representation.

Note: add some notes about:
https://github.com/arvindks/kle/blob/master/covariance/covariance.py

And cite Saibaba's phd thesis about the uncertainty and all.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from time import time
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
from numpy.random import Generator, RandomState
from scipy._lib._util import check_random_state  # To handle random_state
from scipy.linalg import solve
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse.linalg import LinearOperator, eigsh, gmres, lgmres
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

from pyrtid.inverse.regularization.dense import generate_dense_matrix
from pyrtid.inverse.regularization.hmatrix import Hmatrix
from pyrtid.inverse.regularization.toeplitz import create_row, toeplitz_product
from pyrtid.utils.operators import get_super_lu_preconditioner
from pyrtid.utils.types import NDArrayFloat, NDArrayInt


class CallBack:
    """Represents a callback instance."""

    __slots__: List[str] = ["res"]

    def __init__(self) -> None:
        """Initialize the instance."""
        self.res: List[NDArrayFloat] = []

    def __call__(self, rk) -> None:
        self.res.append(rk)

    @property
    def itercount(self) -> int:
        """Return the number of times the callback as been called."""
        return len(self.res)

    def clear(self) -> None:
        """Delete all results."""
        self.res = []


class CovarianceMatrix(LinearOperator):
    """
    Represents a covariance matrix.

    This is an abstract class.
    """

    __slots__: List[str] = ["kernel", "pts", "nugget", "dtype", "count", "solvmatvecs"]

    def __init__(
        self, pts: NDArrayFloat, kernel: Callable, nugget: float = 0.0
    ) -> None:
        """
        Initialize the instance.

        Parameters
        ----------
        pts : NDArrayFloat
            _description_
        kernel : Callable
            _description_
        nugget : float, optional
            _description_, by default 0.0
        """
        self.kernel: Callable = kernel
        self.pts: NDArrayFloat = pts
        self.nugget: float = nugget
        # counters
        self.count: int = 0
        self.solvmatvecs: int = 0
        super().__init__(dtype="d", shape=(self.number_pts, self.number_pts))

    @property
    def number_pts(self) -> int:
        """Number of points in the domain (n)."""
        return self.pts.shape[0]

    def reset_comptors(self) -> None:
        """Set the comptors to zero."""
        self.count = 0
        self.solvmatvecs = 0

    def itercount(self) -> int:
        """Return the number of counts."""
        return self.count

    @abstractmethod
    def solve(self, b: NDArrayFloat) -> NDArrayFloat:
        """Solve Ax = b, with A, the current covariance matrix instance."""

    def get_inv_cov_times_vector(self, x: NDArrayFloat) -> NDArrayFloat:
        """Return $Q^{-1} x$."""
        return self.solve(x)

    @abstractmethod
    def get_diagonal(self) -> NDArrayFloat:
        """Return the diagonal entries of the matrix (variances)."""

    @abstractmethod
    def get_trace(self) -> NDArrayFloat:
        """Return the trace of the covariance matrix."""


def build_preconditioner(
    pts: NDArrayFloat, kernel: Callable, k: int = 100
) -> csr_matrix:
    """
    Implementation of the preconditioner based on changing basis.

    Parameters
    ----------
    pts : NDArrayFloat
        The points (n, m) with n the number of data points and m the dimension of
        coordinates.
    k : int, optional
        Number of local centers in the preconditioner. Controls the sparity of
        the preconditioner. By default 100.

    Returns
    -------
    csr_matrix
        _description_

    Raises
    ------
    ValueError
        _description_

    Notes:
    ------
    Implementation of the preconditioner based on local centers.
    The parameter k controls the sparsity and the effectiveness of the preconditioner.
    Larger k is more expensive but results in fewer iterations.
    For large ill-conditioned systems, it was best to use a nugget effect to make the
    problem better conditioned.
    To Do: implementation based on local centers and additional points. Will remove the
    hack of using nugget effect.

    """

    # Build the tree
    start: float = time()
    tree: cKDTree = cKDTree(pts, leafsize=32)
    end: float = time()
    nb_pts: int = pts.shape[0]
    if nb_pts <= 0:
        raise ValueError("The number of points cannot be null !")
    if nb_pts < k:
        raise ValueError("k must be superior to the number of points !")

    logging.log(logging.INFO, f"Tree building time = {end-start}")

    # Find the nearest neighbors of all the points
    start = time()
    _dist, ind = tree.query(pts, k=k)
    end = time()

    logging.log(logging.INFO, f"Nearest neighbor computation time = {end-start}")

    Q = np.zeros((k, k), dtype="d")
    y = np.zeros((k, 1), dtype="d")

    row = np.tile(np.arange(nb_pts), (k, 1)).transpose()
    col = np.copy(ind)
    nu = np.zeros((nb_pts, k), dtype="d")

    y[0] = 1.0
    start = time()
    for i in np.arange(nb_pts):
        Q = kernel(cdist(pts[ind[i, :], :], pts[ind[i, :], :]))
        nui = np.linalg.solve(Q, y)
        nu[i, :] = np.copy(nui.transpose())
    end = time()

    logging.log(logging.INFO, "Elapsed time = %g" % (end - start))

    ij = np.zeros((nb_pts * k, 2), dtype="i")
    ij[:, 0] = np.copy(np.reshape(row, nb_pts * k, order="F").transpose())
    ij[:, 1] = np.copy(np.reshape(col, nb_pts * k, order="F").transpose())

    data = np.copy(np.reshape(nu, nb_pts * k, order="F").transpose())
    return csr_matrix((data, ij.transpose()), shape=(nb_pts, nb_pts), dtype="d")


class DenseCovarianceMatrix(CovarianceMatrix):
    """Represents a dense covariance matrix."""

    def __init__(
        self,
        pts: NDArrayFloat,
        kernel: Callable,
        len_scale: NDArrayFloat,
        nugget: float = 0,
    ) -> None:
        super().__init__(pts, kernel, nugget)
        self.mat = generate_dense_matrix(pts, kernel, len_scale)

    def _matvec(self, x: NDArrayFloat) -> NDArrayFloat:
        """Return the covariance matrix times the vector x."""
        return np.dot(self.mat, x) * (1 + self.nugget)

    def _rmatvec(self, x: NDArrayFloat) -> NDArrayFloat:
        """Return the covariance matrix conjugate transpose times the vector x."""
        return np.dot(self.mat.T, x)

    def solve(self, b: NDArrayFloat) -> NDArrayFloat:
        """Solve Ax = b, with A, the current covariance matrix instance."""
        return solve(self.mat, b, assume_a="sym")

    def get_diagonal(self) -> NDArrayFloat:
        """Return the diagonal entries of the matrix (variances)."""
        return self.mat.diagonal()

    def get_trace(self) -> NDArrayFloat:
        """Return the trace of the covariance matrix."""
        return self.mat.trace()


class FFTCovarianceMatrix(CovarianceMatrix):
    """
    Represents a fast fourier transform covariance matrix.

    FFT based operations if kernel is stationary or translation invariant and points
    are on a regular grid.
    """

    def __init__(
        self,
        kernel,
        mesh_dim: Union[NDArrayInt, Tuple[float, float]],
        domain_shape: Union[NDArrayInt, Tuple[int, int]],
        len_scale: NDArrayFloat,
        nugget: float = 0.0,
        k: int = 100,
    ) -> None:
        """_summary_

        Parameters
        ----------
        kernel : _type_
            _description_
        mesh_dim : Union[NDArrayInt, Tuple[float, float]]
            _description_
        domain_shape : Union[NDArrayInt, Tuple[int, int]]
            _description_
        len_scale : NDArrayFloat
            _description_
        nugget : float, optional
            _description_, by default 0.0
        k : int, optional
            Number of local centers in the preconditioner. Controls the sparity of
            the preconditioner. It should be inferior to the number of points.
            By default 100.
        """
        self.param_shape: NDArrayInt = np.array(domain_shape, dtype=np.int8)
        self.row, pts = create_row(
            np.array(mesh_dim, dtype=np.int8), self.param_shape, kernel, len_scale
        )
        super().__init__(pts, kernel, nugget)
        self.preconditioner: csr_matrix = build_preconditioner(pts, kernel, k=k)

    def _matvec(self, x: NDArrayFloat) -> NDArrayFloat:
        """Return the covariance matrix times the vector x."""
        return toeplitz_product(x, self.row, self.param_shape) * (1 + self.nugget)

    def _rmatvec(self, x: NDArrayFloat) -> NDArrayFloat:
        """Return the covariance matrix conjugate transpose times the vector x."""
        return toeplitz_product(x, self.row, self.number_pts)

    def solve(
        self, b: NDArrayFloat, tol: float = 1e-12, maxiter: int = 1000
    ) -> NDArrayFloat:
        """Solve Ax = b, with A, the current covariance matrix instance."""
        residual = CallBack()
        x, info = lgmres(
            self,
            b,
            tol=tol,
            maxiter=maxiter,
            callback=residual,
            M=self.preconditioner,
        )
        self.solvmatvecs += residual.itercount
        return x

    def get_diagonal(self) -> NDArrayFloat:
        """Return the diagonal entries of the matrix (variances)."""
        return self.kernel(np.zeros(len(self.pts)))

    def get_trace(self) -> NDArrayFloat:
        """Return the trace of the covariance matrix."""
        return np.sum(self.get_diagonal())


class HCovarianceMatrix(CovarianceMatrix):
    """
    Represents a hierarchical covariance matrix.

    Works for arbitrary kernels on irregular grids
    """

    def __init__(
        self,
        kernel: Callable,
        pts: NDArrayFloat,
        len_scale: NDArrayFloat,
        rkmax: int = 32,
        eps: float = 1.0e-9,
        nugget: float = 0.0,
        is_verbose: bool = False,
        k: int = 100,
    ) -> None:
        n: int = np.size(pts, 0)
        ind = np.arange(n)

        self.H = Hmatrix(pts, kernel, ind, is_verbose, rkmax, eps)

        super().__init__(pts, kernel, nugget)
        self.is_verbose = is_verbose
        self.preconditioner: csr_matrix = build_preconditioner(pts, kernel, k=k)

    def _matvec(self, x: NDArrayFloat) -> NDArrayFloat:
        """Return the covariance matrix times the vector x."""
        y = np.zeros_like(x, dtype="d")
        return self.H.mult(x, y, self.is_verbose) * (1 + self.nugget)

    def _rmatvec(self, x: NDArrayFloat) -> NDArrayFloat:
        """Return the covariance matrix conjugate transpose times the vector x."""
        y = np.zeros_like(x, dtype="d")
        return self.H.transpmult(x, y, self.is_verbose)

    def solve(
        self, b: NDArrayFloat, tol: float = 1e-12, maxiter: int = 1000
    ) -> NDArrayFloat:
        """Solve Ax = b, with A, the current covariance matrix instance."""
        residual = CallBack()
        x, info = lgmres(
            self,
            b,
            tol=tol,
            maxiter=maxiter,
            callback=residual,
            M=self.preconditioner,
        )
        self.solvmatvecs += residual.itercount
        return x

    def get_diagonal(self) -> NDArrayFloat:
        """Return the diagonal entries of the matrix (variances)."""
        return self.kernel(np.zeros(len(self.pts)))

    def get_trace(self) -> NDArrayFloat:
        """Return the trace of the covariance matrix."""
        return np.sum(self.get_diagonal())


class CovarianceMatrixbyUd(CovarianceMatrix):
    """Compressed version of the covariance matrix."""

    def __init__(
        self,
        prior_d: NDArrayFloat,
        prior_u: NDArrayFloat,
        pts: NDArrayFloat,
        kernel: Callable,
        nugget: float = 0.0,
    ) -> None:
        """Initialize the instance."""
        super().__init__(pts, kernel, nugget)
        self.prior_d: NDArrayFloat = prior_d
        self.prior_u: NDArrayFloat = prior_u

    def _matvec(self, x: NDArrayFloat) -> NDArrayFloat:
        """Return the covariance matrix times the vector x."""
        return np.dot(
            self.prior_u,
            np.multiply(self.prior_d, (np.dot(self.prior_u.T, x.reshape(-1, 1)))),
        )

    def _rmatvec(self, x: NDArrayFloat) -> NDArrayFloat:
        """Return the covariance matrix conjugate transpose times the vector x."""
        return self._matvec(x)

    def get_inv_cov_times_vector(self, x: NDArrayFloat) -> NDArrayFloat:
        """Return $Q^{-1} x = ZD^{-1}Z^{T}x$."""
        # np.dot(invZs.T, invZs)
        # Note: x must be a column vector
        return np.dot(
            self.prior_u,
            np.multiply(1.0 / self.prior_d, np.dot(self.prior_u.T, x.reshape(-1, 1))),
        )

    def get_diagonal(self) -> NDArrayFloat:
        """Return the diagonal entries of the matrix (variances)."""
        raise NotImplementedError()

    def get_trace(self) -> NDArrayFloat:
        """Return the trace of the covariance matrix."""
        raise NotImplementedError()


class SparseInvCovarianceMatrix(CovarianceMatrix):
    """
    Represents a sparse inverse covariance matrix.

    Works for arbitrary kernels on irregular grids.
    """

    __slots__ = ["inv_mat", "preconditioner"]

    def __init__(
        self,
        inv_mat: csc_matrix,
    ) -> None:
        """
        Initialize the instance.

        Parameters
        ----------
        inv_mat : csc_matrix
            Sparse precision matrix (inverse of the covariance matrix).
        """
        self.inv_mat: csc_matrix = inv_mat
        self.preconditioner = get_super_lu_preconditioner(self.inv_mat)
        super().__init__(np.array([]), lambda x: x, 0.0)

    @property
    def number_pts(self) -> int:
        """Number of points in the domain (n)."""
        return self.inv_mat.shape[0]

    def _matvec(self, x: NDArrayFloat) -> NDArrayFloat:
        """Return the covariance matrix times the vector x."""
        return self.solve(x)

    def _rmatvec(self, x: NDArrayFloat) -> NDArrayFloat:
        """Return the covariance matrix conjugate transpose times the vector x."""
        return self.solve(x)

    def get_inv_cov_times_vector(self, x: NDArrayFloat) -> NDArrayFloat:
        """Return $Q^{-1} x."""
        return self.inv_mat.dot(x)

    def solve(
        self, b: NDArrayFloat, tol: float = 1e-8, maxiter: int = 100
    ) -> NDArrayFloat:
        """Solve Ax = b, with A, the current covariance matrix instance."""
        residual = CallBack()
        x, info = gmres(
            self.inv_mat,
            b,
            tol=tol,
            maxiter=maxiter,
            callback=residual,
            atol=tol,
            M=self.preconditioner,
        )
        self.solvmatvecs += residual.itercount
        return x

    def get_diagonal(self) -> NDArrayFloat:
        """Return the diagonal entries of the matrix (variances)."""
        raise NotImplementedError

    def get_trace(self) -> NDArrayFloat:
        """Return the trace of the covariance matrix."""
        return np.sum(1 / self.inv_mat.get_diagonal())


def get_prior_eigen_factorization(
    cov_mat: CovarianceMatrix,
    n_pc: int,
    method: str = "arpack",
    random_state: Optional[Union[int, RandomState, Generator]] = None,
) -> Tuple[NDArrayFloat, NDArrayFloat]:
    """
    Compute Eigenmodes of Prior Covariance.

    Parameters
    ----------
    cov_mat : CovarianceMatrix
        The covariance matrix instance to decompose.
    n_pc : int
        Number of principal component in the matrix.
    method : str, optional
        Method used for the decomposition. Only arpack is supported for now,
        by default "arpack".
    random_state: Optional[Union[int, np.random.Generator, np.random.RandomState]]
        Pseudorandom number generator state used to generate resamples.
        If `random_state` is ``None`` (or `np.random`), the
        `numpy.random.RandomState` singleton is used.
        If `random_state` is an int, a new ``RandomState`` instance is used,
        seeded with `random_state`.
        If `random_state` is already a ``Generator`` or ``RandomState``
        instance then that instance is used.

    Raises
    ------
    NotImplementedError
        If a method difference from arpack is used for decomposition.

    Returns
    -------
    Tuple[NDArrayFloat, NDArrayFloat]
        Eigen values and eigen vectors.
    """
    logging.info("Eigendecomposition of Prior Covariance")

    # twopass = False if not 'twopass' in self.params else self.params['twopass']
    start = time()

    # Random state for v0 vector used by eigsh and svds
    if random_state is not None:
        random_state = check_random_state(random_state)
        v0 = random_state.uniform(size=(cov_mat.shape[0],))
    else:
        v0 = None

    if method == "arpack":
        prior_d, prior_u = eigsh(cov_mat, k=n_pc, v0=v0)
        prior_d = prior_d[::-1]
        prior_d = prior_d.reshape(-1, 1)  # make a column vector
        prior_u = prior_u[:, ::-1]
    else:
        raise NotImplementedError

    logging.info(
        "- time for eigendecomposition with k = %d is %g sec"
        % (n_pc, round(time() - start))
    )

    if (prior_d > 0).sum() < n_pc:
        n_pc = (prior_d > 0).sum()
        prior_d = prior_d[:n_pc, :]
        prior_u = prior_u[:, :n_pc]
        logging.warning("Warning: n_pc changed to %d for positive eigenvalues" % (n_pc))

    logging.info(
        "- 1st eigv : %g, %d-th eigv : %g, ratio: %g"
        % (prior_d[0], n_pc, prior_d[-1], prior_d[-1] / prior_d[0])
    )
    return prior_d, prior_u


def cov_mat_to_ud_mat(
    cov_mat: CovarianceMatrix,
    n_pc: int,
    method: str = "arpack",
    random_state: Optional[Union[int, RandomState, Generator]] = None,
) -> CovarianceMatrixbyUd:
    """
    Convert any covariance matrix to a Ud matrix by eigen decomposition.

    Parameters
    ----------
    cov_mat : CovarianceMatrix
        The covariance matrix instance to decompose.
    n_pc : int
        Number of principal component in the matrix.
    method : str, optional
        Method used for the decomposition. Only arpack is supported for now,
        by default "arpack".
    random_state: Optional[Union[int, np.random.Generator, np.random.RandomState]]
        Pseudorandom number generator state used to generate resamples.
        If `random_state` is ``None`` (or `np.random`), the
        `numpy.random.RandomState` singleton is used.
        If `random_state` is an int, a new ``RandomState`` instance is used,
        seeded with `random_state`.
        If `random_state` is already a ``Generator`` or ``RandomState``
        instance then that instance is used.

    Returns
    -------
    CovarianceMatrixbyUd
        Decomposed matrix instance.
    """
    if isinstance(cov_mat, CovarianceMatrixbyUd):
        return cov_mat
    prior_d, prior_u = get_prior_eigen_factorization(
        cov_mat, n_pc, method, random_state
    )
    return CovarianceMatrixbyUd(prior_d, prior_u, cov_mat.pts, cov_mat.kernel)


def get_explained_var(
    eigval: NDArrayFloat,
    cov_mat: Optional[CovarianceMatrix] = None,
    trace_cov_mat: Optional[float] = None,
) -> NDArrayFloat:
    """Return the variance explained by each eigen value."""
    if trace_cov_mat is not None:
        return eigval / trace_cov_mat
    if cov_mat is not None:
        return eigval / cov_mat.get_trace()
    else:
        raise ValueError("You must provide a Covariance matrix instance or the trace !")
