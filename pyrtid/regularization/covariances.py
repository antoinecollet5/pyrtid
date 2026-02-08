# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 Antoine COLLET

"""Provide covariance matrix representation.

Note: add some notes about:
https://github.com/arvindks/kle/blob/master/covariance/covariance.py

And cite Saibaba's phd thesis about the uncertainty and all.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from time import time
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy as sp
from numpy.random import Generator, RandomState
from scipy.linalg import solve
from scipy.sparse import csc_array, csr_array
from scipy.sparse.linalg import LinearOperator, eigsh, gmres, lgmres
from scipy.spatial import cKDTree, distance_matrix
from scipy.spatial.distance import cdist

from pyrtid.regularization.hmatrix import Hmatrix
from pyrtid.regularization.toeplitz import create_toepliz_first_row, toeplitz_product
from pyrtid.utils import (
    NDArrayFloat,
    NDArrayInt,
    SparseFactor,
    check_random_state,
    get_pts_coords_regular_grid,
    sparse_cholesky,
)
from pyrtid.utils.spde import get_variance


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

    __slots__: List[str] = ["dtype", "count", "solvmatvecs"]

    def __init__(self, shape) -> None:
        """
        Initialize the instance.

        Parameters
        ----------
        shape: Tuple[int, int]
            Shape of the matrix.
        """
        # counters
        self.count: int = 0
        self.solvmatvecs: int = 0
        super().__init__(dtype="d", shape=shape)

    @property
    def number_pts(self) -> int:
        """Number of points in the domain (n)."""
        return self.shape[0]

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

    def get_diagonal(self) -> NDArrayFloat:
        """
        Return the diagonal entries of the matrix (variances).

        The matrix is never built explicitly. Instead the matvec interface is
        used to multiply all column of the identity matrix.
        """
        approx_diag = np.zeros(self.number_pts)
        for i in range(self.number_pts):
            # construct the ith row of the identity matrix
            v = np.zeros(self.number_pts)
            v[i] = 1.0
            approx_diag[i] = self.matvec(v)[i]
        return approx_diag

    def get_trace(self) -> float:
        """Return the trace of the covariance matrix."""
        return float(np.sum(self.get_diagonal()))


class KernelCovarianceMatrix(CovarianceMatrix):
    __slots__: List[str] = ["kernel", "pts", "nugget"]

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
        super().__init__((pts.shape[0], pts.shape[0]))
        self.kernel: Callable = kernel
        self.pts: NDArrayFloat = pts
        self.nugget: float = nugget
        # counters
        self.count: int = 0
        self.solvmatvecs: int = 0


def build_preconditioner(
    pts: NDArrayFloat, kernel: Callable, k: int = 100
) -> csr_array:
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
    csr_array
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
    nb_pts: int = pts.shape[0]
    if nb_pts <= 0:
        raise ValueError("The number of points cannot be null !")
    if nb_pts < k:
        raise ValueError("k must be superior to the number of points !")

    # Build the tree
    start: float = time()
    tree: cKDTree = cKDTree(pts, leafsize=32)
    end: float = time()

    logging.log(logging.INFO, f"Tree building time = {end - start}")

    # Find the nearest neighbors of all the points
    start = time()
    _dist, ind = tree.query(pts, k=k)
    end = time()

    logging.log(logging.INFO, f"Nearest neighbor computation time = {end - start}")

    Q = np.zeros((k, k), dtype="d")
    y = np.zeros((k, 1), dtype="d")

    row = np.tile(np.arange(nb_pts), (k, 1)).transpose()
    col = np.copy(ind)
    nu = np.zeros((nb_pts, k), dtype="d")

    y[0] = 1.0
    start = time()

    # TODO: This is very inefficient and must be re-written
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
    return csr_array((data, ij.transpose()), shape=(nb_pts, nb_pts), dtype="d")


class DenseCovarianceMatrix(CovarianceMatrix):
    """Represents a dense covariance matrix."""

    def __init__(
        self,
        mat: NDArrayFloat,
        nugget: float = 0,
    ) -> None:
        self.mat = mat
        self.nugget = nugget
        super().__init__((mat.shape[0], mat.shape[0]))

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


def generate_dense_matrix(
    pts: NDArrayFloat, kernel: Callable, len_scale: NDArrayFloat, nugget: float = 0.0
) -> DenseCovarianceMatrix:
    """
    Generate a dense matrix.

    Compute O(dim^2) interactions.

    Parameters
    ----------
    pts : NDArrayFloat
        DESCRIPTION.
    kernel : TYPE
        DESCRIPTION.
    len_scale: NDArrayFloat
        DESCRIPTION.

    Returns
    -------
    NDArrayFloat
        The dense matrix.
    """
    # Scale the points coordinates
    scaled_pts = np.array(pts, copy=True)
    for dim in range(scaled_pts.shape[1]):
        scaled_pts[:, dim] /= len_scale[dim]
    return DenseCovarianceMatrix(
        kernel(distance_matrix(scaled_pts, scaled_pts)), nugget=nugget
    )


class EnsembleCovarianceMatrix(CovarianceMatrix):
    r"""
    Represents a covariance matrix as an ensemble of realizations.

    For a given ensemble with shape (:math:`N_{s}`, :math:`N_{e}`), the number of
    points and the number of members in the ensemble respectively, the covariance
    matrix :math:`\mathbf{\Sigma_{ss}}` is approximated from the ensemble
    in the standard way of EnKF
    :cite:p:`evensenDataAssimilationEnsemble2007,aanonsenEnsembleKalmanFilter2009`:

    .. math::
        \mathbf{\Sigma_{ss}} = \frac{1}{N_{e} - 1} \sum_{j=1}^{N_{e}}\left(s_{j} -
        \overline{s}\right)\left(s_{j}
        - \overline{s^{l}} \right)^{T}

    Or by defining a matrix of anomalies
    :math:`\mathbf{A} = \mathbf{S} - \overline{\mathbf{S}}`
    with shape  (:math:`N_{s}`, :math:`N_{e}`):

    .. math::
        \mathbf{\Sigma_{ss}} = \frac{1}{N_{e} - 1} \mathbf{A}^{T}\mathbf{A}

    Note
    ----
    Practically, the dense covariance matrix is never built,
    only the anomalies matrix :math:`\mathbf{A}` is used. The product between the
    inverse of the covariance matrix and a vector
    :math:`\mathbf{x} = \mathbf{\Sigma_{ss}}^{-1}\mathbf{b}`
    is obtained solving the system :math:`\mathbf{A}^{T}\mathbf{Ax} = \mathbf{b}`,
    using gmres, where only anomalies matrix vector products are required.
    """

    def __init__(
        self,
        ensemble: NDArrayFloat,
    ) -> None:
        """
        Initiate the instance.

        Parameters
        ----------
        ensemble : NDArrayFloat
            Ensemble of realization with shape (:math:`N_{s}`, :math:`N_{e}`).
        """
        # on axis 1, the number of parameters
        super().__init__((ensemble.shape[1], ensemble.shape[1]))
        self.ensemble = ensemble

    @property
    def anomalies(self) -> NDArrayFloat:
        """
        Return the matrix of anomalies.

        """
        return self.ensemble - np.mean(self.ensemble, axis=0, keepdims=True)

    @property
    def n_ens(self) -> int:
        """Return the number of members in the ensemble."""
        return self.ensemble.shape[0]

    def _matvec(self, x: NDArrayFloat) -> NDArrayFloat:
        """Return the covariance matrix times the vector x (dot product)."""
        return np.linalg.multi_dot([self.anomalies.T, self.anomalies, x]) / (
            self.n_ens - 1
        )  # type: ignore

    def todense(self) -> NDArrayFloat:
        """
        Return a dense representation of the matrix.
        """
        return self.anomalies.T @ self.anomalies / (self.n_ens - 1)

    def solve(
        self, b: NDArrayFloat, rtol: float = 1e-12, maxiter: int = 1000
    ) -> NDArrayFloat:
        """
        Solve A^{T}Ax = b, with A, the anomalies matrix instance.

        Note that the dense covariance matrix is never built.
        """
        residual = CallBack()
        x, info = gmres(
            self,
            b,
            rtol=rtol,
            maxiter=maxiter,
            callback=residual,
            atol=0.0,
            callback_type="legacy",
        )
        self.solvmatvecs += residual.itercount
        return x

    def get_diagonal(self) -> NDArrayFloat:
        """Return the diagonal entries of the matrix (variances)."""
        return np.sum((self.anomalies**2), axis=0) / (self.n_ens - 1.0)


class FFTCovarianceMatrix(KernelCovarianceMatrix):
    """
    Represents a fast fourier transform covariance matrix.

    FFT based operations if kernel is stationary or translation invariant and points
    are on a regular grid.
    """

    def __init__(
        self,
        kernel,
        mesh_dim: Union[float, NDArrayFloat, Sequence[float]],
        domain_shape: Union[int, NDArrayInt, Sequence[int]],
        len_scale: NDArrayFloat,
        nugget: float = 0.0,
        k: int = 100,
        is_use_preconditioner: bool = False,
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
        is_use_preconditioner: bool
            Whether to build the preconditioner at instance creation and use it to
            solve Ax = b systems. The default is False.
        """
        self.param_shape: NDArrayInt = np.array(domain_shape, dtype=np.int8)
        # Coordinates of the points in the grid with shape (Npts, Ndim)
        pts = get_pts_coords_regular_grid(mesh_dim, self.param_shape)

        self.first_row = create_toepliz_first_row(pts, kernel, len_scale)
        super().__init__(pts, kernel, nugget)
        if is_use_preconditioner:
            self.preconditioner: Optional[csr_array] = build_preconditioner(
                pts, kernel, k=k
            )
        else:
            self.preconditioner = None

    def _matvec(self, x: NDArrayFloat) -> NDArrayFloat:
        """Return the covariance matrix times the vector x."""
        return toeplitz_product(x, self.first_row, self.param_shape) * (1 + self.nugget)

    def solve(
        self, b: NDArrayFloat, rtol: float = 1e-12, maxiter: int = 1000
    ) -> NDArrayFloat:
        """Solve Ax = b, with A, the current covariance matrix instance."""
        residual = CallBack()
        x, info = gmres(
            self,
            b,
            rtol=rtol,
            maxiter=maxiter,
            callback=residual,
            M=self.preconditioner,
            atol=0.0,
            callback_type="legacy",
        )
        self.solvmatvecs += residual.itercount
        return x

    def get_diagonal(self) -> NDArrayFloat:
        """Return the diagonal entries of the matrix (variances)."""
        return self.kernel(np.zeros(len(self.pts)))

    def get_trace(self) -> float:
        """Return the trace (sum of the diagonal) of the covariance matrix."""
        return float(np.sum(self.get_diagonal()))


class HCovarianceMatrix(KernelCovarianceMatrix):
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
        self.preconditioner: csr_array = build_preconditioner(pts, kernel, k=k)

    def _matvec(self, x: NDArrayFloat) -> NDArrayFloat:
        """Return the covariance matrix times the vector x."""
        y = np.zeros_like(x, dtype="d")
        return self.H.mult(x, y, self.is_verbose) * (1 + self.nugget)

    def _rmatvec(self, x: NDArrayFloat) -> NDArrayFloat:
        """Return the covariance matrix conjugate transpose times the vector x."""
        y = np.zeros_like(x, dtype="d")
        return self.H.transpmult(x, y, self.is_verbose)

    def solve(
        self, b: NDArrayFloat, rtol: float = 1e-12, maxiter: int = 1000
    ) -> NDArrayFloat:
        """Solve Ax = b, with A, the current covariance matrix instance."""
        residual = CallBack()
        x, info = lgmres(
            self,
            b,
            rtol=rtol,
            maxiter=maxiter,
            callback=residual,
            M=self.preconditioner,
            atol=0.0,
        )
        self.solvmatvecs += residual.itercount
        return x

    def get_diagonal(self) -> NDArrayFloat:
        """Return the diagonal entries of the matrix (variances)."""
        return self.kernel(np.zeros(len(self.pts)))

    def get_trace(self) -> float:
        """Return the trace of the covariance matrix."""
        return float(np.sum(self.get_diagonal()))


class EigenFactorizedCovarianceMatrix(CovarianceMatrix):
    """Compressed version of the covariance matrix."""

    def __init__(
        self,
        eig_vals: NDArrayFloat,
        eig_vects: NDArrayFloat,
    ) -> None:
        """
        Initialize the instance.

        Parameters
        ----------
        eig_vals : NDArrayFloat
            1D vector of eigen values with size `n_pc`.
        eig_vects : NDArrayFloat
            2D arrays of eigen vectors (columns) with size `(Ns, n_pc)`. Ns being the
            number of elements in the original covariance matrix.
        """
        super().__init__((eig_vects.shape[0], eig_vects.shape[0]))
        self.eig_vals: NDArrayFloat = eig_vals
        self.eig_vects: NDArrayFloat = eig_vects

        assert self.eig_vals.size == self.eig_vects.shape[1]

    @property
    def n_pc(self) -> int:
        """
        Return the number of eigen vectors/values, i.e. principal components.

        It is determined from the eigen values vector size.
        """
        return self.eig_vals.size

    def _matvec(self, x: NDArrayFloat) -> NDArrayFloat:
        """Return the covariance matrix times the vector x."""
        return np.dot(
            self.eig_vects,
            np.multiply(self.eig_vals, np.dot(self.eig_vects.T, x.reshape(-1, 1))),
        )

    def solve(self, x: NDArrayFloat) -> NDArrayFloat:
        r"""
        Return $Q^{-1} x = ZD^{-1}Z^{T}x$.

        Parameters
        ----------
        x: NDArrayFloat
            Column vector with shape ($N_{\mathrm{s}}$, 1) or ensemble matrix with
            shape ($N_{\mathrm{s}}$, $N_e$).

        Returns
        -------
        NDAarrayFloat
            Column vector with shape ($N_{\mathrm{s}}$, 1) or ensemble matrix with
            shape ($N_{\mathrm{s}}$, $N_e$).
        """
        # np.dot(invZs.T, invZs)
        # Note: x must be a column vector of a matrix with size (Ns, Ne)
        ne = 1  # case of a column vector
        if x.ndim > 1:
            ne = x.shape[1]
        return np.dot(
            self.eig_vects,
            np.multiply(
                1.0 / self.eig_vals, np.dot(self.eig_vects.T, x.reshape(-1, ne))
            ),
        )

    def get_trace(self) -> float:
        """Return the trace of the covariance matrix."""
        return float(np.sum(self.eig_vals))

    def todense(self) -> NDArrayFloat:
        return np.dot(self.eig_vects, np.multiply(self.eig_vals, self.eig_vects.T))

    def get_sparse_LLT_factor(self) -> csc_array:
        """
        Return the sparse factor L of the LL^T factorization of the eigen matrix.

        Return
        ------
        L: csc_array
            L = U * V^{T/2}.
        """
        # 1) Convert U sqrt(V) to a sparse format
        sp_mat = sp.sparse.lil_array(self.eig_vects * np.sqrt(self.eig_vals).T)
        # 2) Resize -> we now have a square matrix and indices are preserved
        sp_mat.resize(self.shape)
        # 3) Convert to column format
        return sp_mat.tocsc()


class SparseInvCovarianceMatrix(CovarianceMatrix):
    """
    Represents a sparse inverse covariance matrix.

    Works for arbitrary kernels on irregular grids.
    """

    __slots__ = ["inv_mat", "inv_mat_cho_factor", "preconditioner"]

    def __init__(
        self,
        inv_mat: csc_array,
        inv_mat_cho_factor: Optional[SparseFactor] = None,
    ) -> None:
        """
        Initialize the instance.

        Parameters
        ----------
        inv_mat : csc_array
            Sparse precision matrix (inverse of the covariance matrix).
        inv_mat_cho_factor: Optional[SparseFactor]
            inv_mat CHOLMOD Factor. If not provided, the factorization is performed
            at the instance initialization. The default is None.
        """
        self.inv_mat: csc_array = csc_array(inv_mat)

        if inv_mat_cho_factor is None:
            self.inv_mat_cho_factor: SparseFactor = sparse_cholesky(self.inv_mat)
        else:
            self.inv_mat_cho_factor: SparseFactor = inv_mat_cho_factor
        super().__init__(inv_mat.shape)

    def _matvec(self, x: NDArrayFloat) -> NDArrayFloat:
        """Return the covariance matrix times the vector x."""
        return self.inv_mat_cho_factor(x)

    def solve(self, x: NDArrayFloat) -> NDArrayFloat:
        """Return $Q^{-1} x."""
        return self.inv_mat.dot(x)

    def get_diagonal(self) -> NDArrayFloat:
        """
        Return the diagonal entries of the matrix (variances).

        The matrix is never built explicitly. Instead the matvec interface is
        used to multiply all column of the identity matrix.
        """
        return get_variance(self.inv_mat, self.inv_mat_cho_factor)


def get_matrix_eigen_factorization(
    cov_mat: CovarianceMatrix,
    n_pc: int,
    random_state: Optional[Union[int, RandomState, Generator]] = None,
) -> Tuple[NDArrayFloat, NDArrayFloat]:
    """
    Compute Eigenmodes of the covariance.

    Parameters
    ----------
    cov_mat : CovarianceMatrix
        The covariance matrix instance to decompose.
    n_pc : int
        Number of principal component in the matrix.
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

    eig_vals, eig_vects = eigsh(cov_mat, k=n_pc, v0=v0)
    eig_vals = eig_vals[::-1]
    eig_vals = eig_vals.reshape(-1, 1)  # make a column vector
    eig_vects = eig_vects[:, ::-1]

    logging.info(
        "- time for eigendecomposition with k = %d is %g sec"
        % (n_pc, round(time() - start))
    )

    if (eig_vals > 0).sum() < n_pc:
        n_pc = (eig_vals > 0).sum()
        eig_vals = eig_vals[:n_pc, :]
        eig_vects = eig_vects[:, :n_pc]
        logging.warning("Warning: n_pc changed to %d for positive eigenvalues" % (n_pc))

    logging.info(
        f"- 1st eigv : {eig_vals[0]}, {n_pc}-th eigv : {eig_vals[-1]}, "
        f"ratio: {eig_vals[-1] / eig_vals[0]}"
    )
    return eig_vals, eig_vects


def eigen_factorize_cov_mat(
    cov_mat: CovarianceMatrix,
    n_pc: int,
    random_state: Optional[Union[int, RandomState, Generator]] = None,
) -> EigenFactorizedCovarianceMatrix:
    """
    Return an eigen factorized covariance matrix from the input covariance matrix.

    Parameters
    ----------
    cov_mat : CovarianceMatrix
        The covariance matrix instance to decompose.
    n_pc : int
        Number of principal component in the matrix.
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
    EigenFactorizedCovarianceMatrix
        Decomposed matrix instance.
    """
    if isinstance(cov_mat, EigenFactorizedCovarianceMatrix):
        return cov_mat
    eig_vals, eig_vects = get_matrix_eigen_factorization(cov_mat, n_pc, random_state)
    return EigenFactorizedCovarianceMatrix(eig_vals, eig_vects)


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


def sample_from_sparse_cov_factor(
    mean: NDArrayFloat,
    factor: csc_array,
    n_samples: int = 100,
    random_state: Optional[
        Union[int, np.random.Generator, np.random.RandomState]
    ] = None,
) -> NDArrayFloat:
    r"""
    Sample from the given sparse factor of the covariance matrix and the given mean.

    Parameters
    ----------
    mean: NDArrayFloat
        Mean of the field with shape $N_{\mathrm{s}}$.
    factor: NDArrayFloat
        Sparse factor of the covariance matrix from which to sample from. It has shape
        $(N_{\mathrm{s}} \times N_{\mathrm{s}})$.
    n_samples: int
        The number of samples required ($N_{\mathrm{e}}$). By default 100.
    random_state : Optional[Union[int, np.random.Generator, np.random.RandomState]]
        Pseudorandom number generator state used to generate resamples.
        If `random_state` is ``None`` (or `np.random`), the
        `numpy.random.RandomState` singleton is used.
        If `random_state` is an int, a new ``RandomState`` instance is used,
        seeded with `random_state`.
        If `random_state` is already a ``Generator`` or ``RandomState``
        instance then that instance is used. The default is None

    Returns
    -------
    NDArrayFloat
        The ensemble of realizations with shape
        $(N_{\mathrm{s}} \times N_{\mathrm{e}})$
    """
    if random_state is not None:
        _random_state = check_random_state(random_state)
    else:
        _random_state = np.random.default_rng()
    return factor @ _random_state.normal(
        scale=1.0, size=(factor.shape[0], n_samples)
    ) + mean.reshape(-1, 1)
