"""Implement functions to perform geostatistics from the spde approach."""

import math
from typing import Optional, Union

import numpy as np
import scipy as sp
from scipy._lib._util import check_random_state  # To handle random_state
from scipy.sparse import csc_array, lil_array
from sksparse.cholmod import Factor

from pyrtid.utils import sparse_cholesky
from pyrtid.utils.grid import indices_to_node_number, span_to_node_numbers_3d
from pyrtid.utils.types import NDArrayFloat, NDArrayInt


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
    cholQ: Factor,
    w: Optional[NDArrayFloat] = None,
    random_state: Optional[
        Union[int, np.random.Generator, np.random.RandomState]
    ] = None,
) -> NDArrayFloat:
    """
    Return a non conditional simulation for the given precision matrix factorization.

    Parameters
    ----------
    cholQ : Factor
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
        if random_state is not None:
            random_state = check_random_state(random_state)
        else:
            random_state = np.random.default_rng()
        w = random_state.normal(size=cholQ.L().shape[0])  # white noise

    # Note: https://scikit-sparse.readthedocs.io/en/latest/cholmod.html
    # We want to solve A z = w  ==> z = A^{-1} w
    # We use the cholesky factorization LDL' = PA'AP'
    # with P' = P^{-1} the permutation that makes the decomposition unique.
    # So LD^{1/2} = PA' and A = D^{1/2}L'P
    # Finally z = P'L^{-T}D^{-1/2} w
    # Possible to do it with gmres -> not efficient without preconditioner
    # from scipy.sparse.linalg import gmres
    # L, D = cholQ.L_D()
    # z, _ = cholQ.apply_Pt(
    #    gmres(L.T.tocsc(), 1.0 / np.sqrt(D.diagonal()) * w, tol=1e-12)
    # )
    # CHOLMOD is the most performant
    return cholQ.apply_Pt(cholQ.solve_Lt(1.0 / np.sqrt(cholQ.D()) * w))


def simu_nc_t(
    cholQ: Factor,
    w: Optional[NDArrayFloat] = None,
    random_state: Optional[
        Union[int, np.random.Generator, np.random.RandomState]
    ] = None,
) -> NDArrayFloat:
    """
    Return the transpose operator of :func:`simu_nc`.

    Parameters
    ----------
    cholQ : Factor
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
        if random_state is not None:
            random_state = check_random_state(random_state)
        else:
            random_state = np.random.default_rng()
        w = random_state.normal(size=cholQ.L().shape[0])  # white noise

    # Note: https://scikit-sparse.readthedocs.io/en/latest/cholmod.html
    # We want to solve z = A^{-T} w
    # since A = D^{1/2}L'P  (see simu_nc)
    # A^{-1} = P'L^{-T}D^{-1/2}
    # z = D^{1/2}L^{-1} P w
    return 1.0 / np.sqrt(cholQ.D()) * cholQ.solve_L(cholQ.apply_P(w))


def simu_nc_t_inv(
    cholQ: Factor,
    z: NDArrayFloat,
) -> NDArrayFloat:
    """
    Return the inverse of the transpose operator :func:`simu_nc_t` of :func:`simu_nc`.

    Parameters
    ----------
    cholQ : Factor
        The cholesky factorization of precision matrix Q.
    z: NDArrayFloat
        Input vector (results of :func:`simu_nc_t`.)

    Returns
    -------
    NDArrayFloat
        Input of :func:`simu_nc` that caused the input z.


    """
    # Note: https://scikit-sparse.readthedocs.io/en/latest/cholmod.html
    # We want to w = A^{T} z
    # z = D^{1/2}L^{-1} P w (see simu_nc_t)
    # So w = P^{-1}LD^{1/2} z
    return cholQ.apply_Pt(cholQ.L_D()[0] @ (np.sqrt(cholQ.D()) * z))


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
    Q_cond: csc_array,
    dat: NDArrayFloat,
    dat_indices: NDArrayInt,
    cholQ_cond: Optional[Factor] = None,
    dat_var: Optional[NDArrayFloat] = None,
) -> NDArrayFloat:
    """
    Return a krigging.

    Parameters
    ----------
    Q_cond : csc_array
        Conditional precision matrix.
    dat : NDArrayFloat
        Conditional values.
    dat_indices : NDArrayInt
        Grid cell indices of the conditional values. The default is None.
    cholQ_cond : Factor
        Cholesky decomposition of the unconditional precision matrix.
    dat_var : NDArrayFloat
        Variance of the conditional data. The default is None.

    Returns
    -------
    NDArrayFloat
        Krigging.
    """
    if cholQ_cond is None:
        _cholQ_cond = sparse_cholesky((Q_cond.tocsc()))
    else:
        _cholQ_cond = cholQ_cond

    input = np.zeros(Q_cond.shape[0])
    input[dat_indices] = dat
    if dat_var is not None:
        input[dat_indices] /= dat_var

    # An alternative to build the input vector is to use a sparse matrix
    # I write it there because it is required when transposing the krigging operator.
    # Z = lil_array((Q.shape[0], dat_indices.size))
    # Z[dat_indices, np.arange(dat_indices.size)] = 1
    # input_bis = Z @ dat
    # if dat_var is not None:
    #     input_bis[dat_indices] /= dat_var
    # checking the correctness
    # np.testing.assert_allclose(input, input_bis)
    return _cholQ_cond(input)


def d_simu_nc_mat_vec(cholQ: Factor, b: NDArrayFloat) -> NDArrayFloat:
    """
    Return the product between the derivative of simu_nc and a vector.

    Parameters
    ----------
    cholQ : Factor
        The cholesky factorization of unconditional precision matrix Q.
    b : NDArrayFloat
        Input vector b.

    Returns
    -------
    NDArrayFloat
        Results of the transposed non-conditional simulation operator
        applied to the input vector b.
    """
    return simu_nc_t(cholQ, b)


def d_simu_nc_mat_vec_inv(cholQ: Factor, b: NDArrayFloat) -> NDArrayFloat:
    """
    Return the product between the derivative of simu_nc and a vector.

    Parameters
    ----------
    cholQ : Factor
        The cholesky factorization of unconditional precision matrix Q.

    b : NDArrayFloat
        Input vector b.

    Returns
    -------
    NDArrayFloat
        Results of the inverse-transposed non-conditional simulation operator applied
        to the input vector b.
    """
    return simu_nc_t_inv(cholQ, b)


def simu_c(
    cholQ: Factor,
    Q_cond: csc_array,
    cholQ_cond: Factor,
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
    cholQ : Factor
        Cholesky decomposition of the unconditional precision matrix.
    Q_cond : csc_array
        Conditional precision matrix.
    cholQ_cond : Factor
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
    z_k = kriging(Q_cond, dat, dat_indices, cholQ_cond=cholQ_cond, dat_var=dat_var)
    # z_k = krig_prec2(Q_cond, dat * 1 / grid_var[dat_indices], dat_indices)
    z_nc = simu_nc(cholQ, w, random_state)
    dat_nc = z_nc[dat_indices]
    # z_nck = krig_chol(QTT_factor, QTD, dat_nc, dat_indices)
    z_nck = kriging(Q_cond, dat_nc, dat_indices, cholQ_cond=cholQ_cond, dat_var=dat_var)
    return z_k - (z_nc - z_nck)


def d_simu_c_matvec(
    cholQ: Factor,
    cholQ_cond: Factor,
    dat_indices: NDArrayInt,
    dat_var: NDArrayFloat,
    b: NDArrayFloat,
) -> NDArrayFloat:
    """
    Return the product between the derivative of simu_c and a vector.

    The Jacobian matrix mapping the white noise and the parameter.

    Parameters
    ----------
    cholQ : Factor
        Cholesky decomposition of the unconditional precision matrix.
    Q_cond : csc_array
        Conditional precision matrix.
    cholQ_cond : Factor
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
    Z = lil_array((cholQ_cond.L().shape[0], dat_indices.size))
    Z[dat_indices, np.arange(dat_indices.size)] = 1
    return simu_nc_t(cholQ, Z @ (1 / dat_var * (Z.T @ cholQ_cond(b))) - b)


def get_variance(Q: csc_array, cholQ: Optional[Factor]) -> NDArrayFloat:
    """
    Extract efficiently the diagonal of the covariance matrix from the precision matrix.

    It relies on the linear operator `matvec` operation and consequenlty does not
    require to build the dense matrix which is much longer and generally untractable
    for large-scale problems.

    Parameters
    ----------
    hess_inv : LbfgsInvHessProduct
        Linear operator for the L-BFGS approximate inverse Hessian.

    Returns
    -------
    NDArrayFloat
        The diagonal of the L-BFGS approximated inverse Hessian.
    """
    # perform the cholesky factorization -> solving is then much faster
    if cholQ is None:
        _cholQ = sparse_cholesky(Q.tocsc())
    else:
        _cholQ = cholQ
    n_params = Q.shape[0]
    cov_mat_diag = np.zeros(n_params)
    v = np.zeros(n_params)
    for i in range(n_params):
        v[i - 1] = 0.0
        v[i] = 1.0
        cov_mat_diag[i] = _cholQ(v)[i]
    return cov_mat_diag
