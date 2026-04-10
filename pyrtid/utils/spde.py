# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 Antoine COLLET

"""Implement functions to perform geostatistics from the spde approach."""

import math
from typing import Optional, Union

import covmats
import numpy as np
import scipy as sp
from scipy.sparse import csc_array, lil_array

from pyrtid.utils.grid import indices_to_node_number, span_to_node_numbers_3d
from pyrtid.utils.types import NDArrayFloat, NDArrayInt


def matern_kernel(r: NDArrayFloat, len_scale: float = 1, v: float = 1) -> NDArrayFloat:
    """
    Computes Matérn correlation function for given distances.

    Parameters:
    -----------
    r : array
        Distances between locations.
    len_scale : float
        Range parameter (ϕ). Must be greater than 0.
    v : float
        Smoothness parameter (nu). Must be greater than 0.
    Returns:
    --------
    Array giving Matern correlation for given distances.
    """
    r = np.abs(r)
    r[r == 0] = 1e-8
    return (
        2 ** (1 - v)
        / sp.special.gamma(v)
        * (np.sqrt(2 * v) * r / len_scale) ** v
        * sp.special.kv(v, np.sqrt(2 * v) * r / len_scale)
    )


def get_laplacian_matrix_for_loops(
    nx: int,
    ny: int,
    nz: int,
    dx: float,
    dy: float,
    dz: float,
    kappa: Union[NDArrayFloat, float],
) -> csc_array:
    """
    Return a sparse matrix of the discretization of the Laplacian.

    Note
    ----
    This is a very inefficient implementation which is simply dedicated to check the
    vectorial implementation correctness. This could be interesting when sparse objects
    are supported by numba with the jit compiler.

    Parameters
    ----------
    nx : int
        Number of grid cells along x.
    ny : int
        Number of grid cells along y.
    nz : int
        Number of grid cells along z.
    dx : float
        Size of the mesh along x.
    dy : float
        Size of the mesh along y.
    dz : float
        Size of the mesh along z.
    kappa : float
        Range (length scale).

    Returns
    -------
    csc_array
        Sparse matrix with dimension (nx * ny)x(nx * ny) representing the  discretized
        laplacian.

    """
    n_nodes = nx * ny * nz
    if np.isscalar(kappa):
        _kappa = np.full(n_nodes, fill_value=kappa)
    else:
        _kappa = np.array(kappa).ravel("F")
    # construct an empty sparse matrix (lil_format because it supports indexing and
    # slicing).
    lap = lil_array((n_nodes, n_nodes), dtype=np.float64)

    # Looping on all nodes and considering neighbours
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                node_index = int(indices_to_node_number(ix, nx, iy, ny, iz))
                lap[node_index, node_index] += _kappa[node_index] ** 2

                if nx > 1:
                    lap[node_index, node_index] += 2 / dx**2
                if ny > 1:
                    lap[node_index, node_index] += 2 / dy**2
                if nz > 1:
                    lap[node_index, node_index] += 2 / dz**2

                # X contribution
                if ix > 0:
                    neighbor_index = int(indices_to_node_number(ix - 1, nx, iy, ny, iz))
                    lap[node_index, neighbor_index] += -1.0 / dx**2
                if ix < nx - 1:
                    neighbor_index = int(indices_to_node_number(ix + 1, nx, iy, ny, iz))
                    lap[node_index, neighbor_index] += -1.0 / dx**2

                # Y contribution
                if iy > 0:
                    neighbor_index = int(indices_to_node_number(ix, nx, iy - 1, ny, iz))
                    lap[node_index, neighbor_index] += -1.0 / dy**2
                if iy < ny - 1:
                    neighbor_index = int(indices_to_node_number(ix, nx, iy + 1, ny, iz))
                    lap[node_index, neighbor_index] += -1.0 / dy**2

                # Z contribution
                if iz > 0:
                    neighbor_index = int(indices_to_node_number(ix, nx, iy, ny, iz - 1))
                    lap[node_index, neighbor_index] += -1.0 / dz**2
                if iz < nz - 1:
                    neighbor_index = int(
                        indices_to_node_number(
                            ix,
                            nx,
                            iy,
                            ny,
                            iz + 1,
                        )
                    )
                    lap[node_index, neighbor_index] += -1.0 / dz**2

    # Convert from lil to csr matrix for more efficient calculation
    return lap.tocsc()


def get_laplacian_matrix(
    nx: int,
    ny: int,
    nz: int,
    dx: float,
    dy: float,
    dz: float,
    kappa: Union[NDArrayFloat, float],
) -> csc_array:
    """
    Return a sparse matrix of the discretization of the Laplacian.

    Note
    ----
    This should be a bit more efficient than the for-loop version for large datasets.

    Parameters
    ----------
    nx : int
        Number of grid cells along x.
    ny : int
        Number of grid cells along y.
    nz : int
        Number of grid cells along z.
    dx : float
        Size of the mesh along x.
    dy : float
        Size of the mesh along y.
    dz : float
        Size of the mesh along z.
    kappa : float
        Range (length scale).

    Returns
    -------
    csc_array
        Sparse matrix with dimension (nx * ny)x(nx * ny) representing the  discretized
        laplacian.

    """
    n_nodes = nx * ny * nz
    if np.isscalar(kappa):
        _kappa = np.full(n_nodes, fill_value=kappa)
    else:
        _kappa = np.array(kappa).ravel("F")
    # construct an empty sparse matrix (lil_format because it supports indexing and
    # slicing).
    lap = lil_array((n_nodes, n_nodes), dtype=np.float64)

    # Add kappa on the diagonal
    lap.setdiag(lap.diagonal() + _kappa**2)

    # X contribution
    if nx > 1:
        lap.setdiag(lap.diagonal() + 2 / dx**2)
        indices_owner: NDArrayInt = span_to_node_numbers_3d(
            (slice(0, nx - 1), slice(None), slice(None)), nx=nx, ny=ny, nz=nz
        )
        indices_neigh: NDArrayInt = span_to_node_numbers_3d(
            (slice(1, nx), slice(None), slice(None)), nx=nx, ny=ny, nz=nz
        )

        # forward
        lap[indices_owner, indices_neigh] -= np.ones(indices_owner.size) / dx**2
        # backward
        lap[indices_neigh, indices_owner] -= np.ones(indices_owner.size) / dx**2

    # Y contribution
    if ny > 1:
        lap.setdiag(lap.diagonal() + 2 / dy**2)
        indices_owner: NDArrayInt = span_to_node_numbers_3d(
            (slice(None), slice(0, ny - 1), slice(None)), nx=nx, ny=ny, nz=nz
        )
        indices_neigh: NDArrayInt = span_to_node_numbers_3d(
            (slice(None), slice(1, ny), slice(None)), nx=nx, ny=ny, nz=nz
        )

        # forward
        lap[indices_owner, indices_neigh] -= np.ones(indices_owner.size) / dy**2
        # backward
        lap[indices_neigh, indices_owner] -= np.ones(indices_owner.size) / dy**2

    # Z contribution
    if nz > 1:
        lap.setdiag(lap.diagonal() + 2 / dz**2)
        indices_owner: NDArrayInt = span_to_node_numbers_3d(
            (slice(None), slice(None), slice(0, nz - 1)), nx=nx, ny=ny, nz=nz
        )
        indices_neigh: NDArrayInt = span_to_node_numbers_3d(
            (slice(None), slice(None), slice(1, nz)), nx=nx, ny=ny, nz=nz
        )

        # forward
        lap[indices_owner, indices_neigh] -= np.ones(indices_owner.size) / dz**2
        # backward
        lap[indices_neigh, indices_owner] -= np.ones(indices_owner.size) / dz**2

    # Convert from lil to csr matrix for more efficient calculation
    return lap.tocsc()


def get_precision_matrix(
    nx: int,
    ny: int,
    nz: int,
    dx: float,
    dy: float,
    dz: float,
    kappa: Union[float, NDArrayFloat],
    alpha: float,
    spatial_dim: int,
    sigma: Union[float, NDArrayFloat] = 1.0,
    is_use_mass_lumping: bool = True,
) -> csc_array:
    """
    Get the precision matrix for the given SPDE field parameters.

    Parameters
    ----------
    nx : int
        Number of grid cells along x.
    ny : int
        Number of grid cells along y.
    nz: int
        Number of grid cells along z.
    dx : float
        Size of the mesh along x.
    dy : float
        Size of the mesh along y.
    dz : float
        Size of the mesh along z.
    kappa : NDArrayFloat
        SPDE parameter linked to the inverse of the correlation range of the covariance
        function. Vector of real strictly positive.
    alpha : float
        SPDE parameter linked to the field regularity. 2 * alpha must be an integer.
    spatial_dim : int
        Spatial dimension of the grid (1, 2 or 3).
    sigma: Union[float, NDArrayFloat], optional
        The marginal variance. If it changes throughout the domain, a (nx * ny) 1D array
        is expected. The default is 1.0.
    is_use_mass_lumping: bool
        Approximate the matrix power. The default is True.

    Returns
    -------
    csc_array
        The sparse precision matrix.
    """

    # Check if 2 alpha is an integer
    if alpha < 1.0 or not float(alpha).is_integer():
        raise ValueError(
            "alpha must be superior or equal to 1.0 and must be an whole number!"
        )
    # Discretization of (kappa^2 - Delta)^(alpha)
    # Build the laplacian matrix: (kappa^2 - Delta)
    A: csc_array = get_laplacian_matrix(nx, ny, nz, dx, dy, dz, kappa)

    # Apply alpha (we deal only with integers alpha)
    Af = sp.sparse.identity(A.shape[0])

    # Use mass lumping
    for i in range(int(alpha)):
        # Af = A @ Af  # matrix multiplication
        Af = A @ Af

    # Correction factor for variance
    nu = 2 * alpha - spatial_dim / 2
    tau = (kappa ** (nu)) * np.sqrt(
        (4 * np.pi) * math.gamma(2 * alpha) / math.gamma(nu)
    )

    # Calculate precision matrix
    Af = np.sqrt((dx * dy)) * Af / (tau * sigma)
    return (Af.T @ Af).tocsc()


def simu_nc(
    cov: covmats.CovarianceMatrix,
    w: Optional[NDArrayFloat] = None,
    random_state: Optional[
        Union[int, np.random.Generator, np.random.RandomState]
    ] = None,
) -> NDArrayFloat:
    """
    Return a non conditional simulation for the given precision matrix factorization.

    Parameters
    ----------
    cov: covmats.CovarianceMatrix
        The cholesky factorization of precision matrix.
    w: Optional[NDArrayFloat]
        Gaussian white noise with mean zero and standard deviation 1.0. If not
        provided, the random state is used to generate the noise.
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
        The non conditional simulation.

    """
    # Random state for v0 vector used by eigsh and svds
    if w is None:
        cov.sample_mvnormal(shape=(1,), random_state=random_state)
    return cov.colorize(w)


def simu_nc_t(
    cov: covmats.CovarianceMatrix,
    w: Optional[NDArrayFloat] = None,
    random_state: Optional[
        Union[int, np.random.Generator, np.random.RandomState]
    ] = None,
) -> NDArrayFloat:
    """
    Return the transpose operator of :func:`simu_nc`.

    Parameters
    ----------
    scf : covmats.SparseCholeskyFactor
        The cholesky factorization of precision matrix.
    w: Optional[NDArrayFloat]
        Gaussian white noise with mean zero and standard deviation 1.0. If not
        provided, the random state is used to generate the noise.
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
        The non conditional simulation.

    """
    return simu_nc(cov, w, random_state).T


def simu_nc_t_inv(
    cov: covmats.CovarianceMatrix,
    z: NDArrayFloat,
) -> NDArrayFloat:
    """
    Return the inverse of the transpose operator :func:`simu_nc_t` of :func:`simu_nc`.

    Parameters
    ----------
    cov: covmats.CovarianceMatrix
        The cholesky factorization of precision matrix Q.
    z: NDArrayFloat
        Input vector (results of :func:`simu_nc_t`.)

    Returns
    -------
    NDArrayFloat
        Input of :func:`simu_nc` that caused the input z.


    """
    return cov.whiten(z)


def condition_precision_matrix(
    Q: csc_array, dat_indices: NDArrayInt, dat_var: NDArrayFloat
) -> csc_array:
    """
    Condition the precision matrix with the variance of known data points.

    Parameters
    ----------
    Q : csc_array
        _description_
    dat_indices : NDArrayInt
        _description_
    dat_var : NDArrayFloat
        _description_

    Returns
    -------
    csc_array
        The conditioned precision matrix.
    """
    # Build the diagonal matrix containing the inverse of the error variance at known
    # data points

    diag_var = lil_array(Q.shape)
    diag_var[dat_indices, dat_indices] = 1 / dat_var
    return (diag_var + Q).tocsc()


def kriging(
    cov,  # covmats.CovarianceMatrix
    dat: np.ndarray,
    dat_indices: np.ndarray,
    dat_var: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Matrix-free kriging using a covariance operator.

    Parameters
    ----------
    cov : CovarianceMatrix
        Must implement:
            - cov.matvec(x): returns Σx
            - cov.shape -> (n, n)
    dat : (m,) array
        Observed values
    dat_indices : (m,) int array
        Indices of observations
    dat_var : (m,) array, optional
        Observation noise variance. If None → exact kriging.

    Returns
    -------
    mu : (n,) array
        Kriging posterior mean
    """
    n = cov.shape[0]
    m = len(dat_indices)

    if dat_var is None:
        dat_var = np.zeros(m)

    # --- Step 1: build K = H Σ H^T + R ---
    E = np.zeros((n, m))
    E[dat_indices, np.arange(m)] = 1.0

    SE = np.column_stack([cov.matvec(E[:, j]) for j in range(m)])
    K = SE[dat_indices, :] + np.diag(dat_var)

    # --- Step 2: solve K alpha = dat ---
    # Add tiny nugget for numerical stability if needed
    # K += 1e-10 * np.eye(m)
    # Works fine for a limited number of observations
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, dat))

    # --- Step 3: compute μ = Σ H^T α ---
    v = np.zeros(n)
    v[dat_indices] = alpha

    mu = cov.matvec(v)

    # --- Variance ---
    var = np.zeros(n)

    for i in range(n):
        # Compute k_i
        e = np.zeros(n)
        e[i] = 1.0
        col = cov.matvec(e)

        k_i = col[dat_indices]

        # Solve K^{-1} k_i via Cholesky
        tmp = np.linalg.solve(L, k_i)
        v_i = np.linalg.solve(L.T, tmp)

        # Σ_ii
        sigma_ii = col[i]

        var[i] = sigma_ii - k_i @ v_i

    return mu


# def kriging(
#     cov,
#     dat: np.ndarray,
#     dat_indices: np.ndarray,
#     dat_var: np.ndarray,
#     tol: float = 1e-6,
#     maxiter: int = 1000,
# ):
#     """
#     Scalable kriging using precision formulation + CG.

#     Requirements on cov:
#         - cov.solve(x): solves Σ z = x  → returns z
#           OR
#         - cov.precision_matvec(x): returns Qx = Σ^{-1}x

#     Parameters
#     ----------
#     cov : CovarianceMatrix
#     dat : (m,)
#     dat_indices : (m,)
#     dat_var : (m,)
#     """

#     n = cov.shape[0]

#     inv_var = 1.0 / dat_var

#     # --- RHS: b = H^T R^{-1} y ---
#     b = np.zeros(n)
#     b[dat_indices] = dat * inv_var

#     # --- Define linear operator A x ---
#     def matvec(x):
#         # Σ^{-1} x
#         if hasattr(cov, "precision_matvec"):
#             y = cov.precision_matvec(x)
#         else:
#             # apply Σ^{-1} via solve
#             y = cov.solve(x)

#         # + H^T R^{-1} H x  (diagonal update)
#         y[dat_indices] += inv_var * x[dat_indices]

#         return y

#     # --- Solve A x = b ---
#     mu, info = cg(
#         A=sp.sparse.linalg.LinearOperator((n, n), matvec=matvec),
#         b=b,
#         tol=tol,
#         maxiter=maxiter,
#     )

#     if info != 0:
#         raise RuntimeError(f"CG did not converge: info={info}")

#     return mu


# def kriging(
#     cov: covmats.CovarianceMatrix,
#     dat: NDArrayFloat,
#     dat_indices: NDArrayInt,
#     dat_var: Optional[NDArrayFloat] = None,
# ) -> NDArrayFloat:
#     """
#     Return a krigging.

#     Parameters
#     ----------
#     Q_cond : csc_array
#         Conditional precision matrix.
#     dat : NDArrayFloat
#         Conditional values.
#     dat_indices : NDArrayInt
#         Grid cell indices of the conditional values. The default is None.
#     scf_cond : covmats.SparseCholeskyFactor
#         Cholesky decomposition of the unconditional precision matrix.
#     dat_var : NDArrayFloat
#         Variance of the conditional data. The default is None.

#     Returns
#     -------
#     NDArrayFloat
#         Krigging.
#     """
#     input = np.zeros(Q_cond.shape[0])
#     input[dat_indices] = dat
#     if dat_var is not None:
#         input[dat_indices] /= dat_var

#     # An alternative to build the input vector is to use a sparse matrix
#     # I write it there because it is required when transposing the krigging operator.
#     # Z = lil_array((Q.shape[0], dat_indices.size))
#     # Z[dat_indices, np.arange(dat_indices.size)] = 1
#     # input_bis = Z @ dat
#     # if dat_var is not None:
#     #     input_bis[dat_indices] /= dat_var
#     # checking the correctness
#     # np.testing.assert_allclose(input, input_bis)
#     return scf_cond(input)


def d_simu_nc_mat_vec(cov: covmats.CovarianceMatrix, b: NDArrayFloat) -> NDArrayFloat:
    """
    Return the product between the derivative of simu_nc and a vector.

    Parameters
    ----------
    scf : covmats.SparseCholeskyFactor
        The cholesky factorization of unconditional precision matrix Q.
    b : NDArrayFloat
        Input vector b.

    Returns
    -------
    NDArrayFloat
        Results of the transposed non-conditional simulation operator
        applied to the input vector b.
    """
    return simu_nc_t(cov, b)


def d_simu_nc_mat_vec_inv(
    scf: covmats.SparseCholeskyFactor, b: NDArrayFloat
) -> NDArrayFloat:
    """
    Return the product between the derivative of simu_nc and a vector.

    Parameters
    ----------
    scf : covmats.SparseCholeskyFactor
        The cholesky factorization of unconditional precision matrix Q.

    b : NDArrayFloat
        Input vector b.

    Returns
    -------
    NDArrayFloat
        Results of the inverse-transposed non-conditional simulation operator applied
        to the input vector b.
    """
    return simu_nc_t_inv(scf, b)


def simu_c(
    cov: covmats.CovarianceMatrix,
    Q_cond: csc_array,
    scf_cond: covmats.SparseCholeskyFactor,
    dat: NDArrayFloat,
    dat_indices: NDArrayInt,
    dat_var: NDArrayFloat,
    w: Optional[NDArrayFloat] = None,
    random_state: Optional[
        Union[int, np.random.Generator, np.random.RandomState]
    ] = None,
) -> NDArrayFloat:
    """
    Generate a conditional simulation.

    Parameters
    ----------
    cov: covmats.CovarianceMatrix
        Covariance matrix.
    Q_cond : csc_array
        Conditional precision matrix.
    scf_cond : covmats.SparseCholeskyFactor
        Cholesky factorization of the conditional precision matrix.
    dat : NDArrayFloat
        Conditional values.
    dat_indices : NDArrayInt
        Grid cell indices of the conditional values.
    dat_var : NDArrayFloat
        Variance of the conditional data.
    w: Optional[NDArrayFloat]
        Gaussian white noise with mean zero and standard deviation 1.0. If not
        provided, the random state is used to generate the noise.
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
        Conditional simulation.
    """
    z_k = kriging(Q_cond, dat, dat_indices, scf_cond=scf_cond, dat_var=dat_var)
    # z_k = krig_prec2(Q_cond, dat * 1 / grid_var[dat_indices], dat_indices)
    z_nc = simu_nc(cov, w, random_state)
    dat_nc = z_nc[dat_indices]
    # z_nck = krig_chol(QTT_factor, QTD, dat_nc, dat_indices)
    z_nck = kriging(Q_cond, dat_nc, dat_indices, scf_cond=scf_cond, dat_var=dat_var)
    return z_k - (z_nc - z_nck)


def d_simu_c_matvec(
    scf: covmats.SparseCholeskyFactor,
    scf_cond: covmats.SparseCholeskyFactor,
    dat_indices: NDArrayInt,
    dat_var: NDArrayFloat,
    b: NDArrayFloat,
) -> NDArrayFloat:
    """
    Return the product between the derivative of simu_c and a vector.

    The Jacobian matrix mapping the white noise and the parameter.

    Parameters
    ----------
    scf : covmats.SparseCholeskyFactor
        Cholesky decomposition of the unconditional precision matrix.
    Q_cond : csc_array
        Conditional precision matrix.
    scf_cond : covmats.SparseCholeskyFactor
        Cholesky factorization of the conditional precision matrix.
    dat_indices : NDArrayInt
        Grid cell indices of the conditional values.
    dat_var : NDArrayFloat
        Variance of the conditional data.
    w: Optional[NDArrayFloat]
        Gaussian white noise with mean zero and standard deviation 1.0. If not
        provided, the random state is used to generate the noise.
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
        Conditional simulation.
    """
    Z = lil_array((scf_cond.L().shape[0], dat_indices.size))
    Z[dat_indices, np.arange(dat_indices.size)] = 1
    return simu_nc_t(scf, Z @ (1 / dat_var * (Z.T @ scf_cond(b))) - b)
