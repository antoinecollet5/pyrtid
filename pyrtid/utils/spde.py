"""Implement functions to perform geostatistics from the spde approach."""
from typing import Optional, Union

import numpy as np
import scipy as sp
from scipy._lib._util import check_random_state  # To handle random_state
from scipy.sparse import csc_matrix, lil_matrix
from sksparse.cholmod import Factor, cholesky

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
) -> csc_matrix:
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
        Number of meshes along x.
    ny : int
        Number of meshes along y.
    nz : int
        Number of meshes along z.
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
    csc_matrix
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
    lap = lil_matrix((n_nodes, n_nodes), dtype=np.float64)

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
) -> csc_matrix:
    """
    Return a sparse matrix of the discretization of the Laplacian.

    Note
    ----
    This should be a bit more efficient than the for-loop version for large datasets.

    Parameters
    ----------
    nx : int
        Number of meshes along x.
    ny : int
        Number of meshes along y.
    nz : int
        Number of meshes along z.
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
    csc_matrix
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
    lap = lil_matrix((n_nodes, n_nodes), dtype=np.float64)

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
) -> csc_matrix:
    """
    Get the precision matrix for the given SPDE field parameters.

    Parameters
    ----------
    nx : int
        Number of meshes along x.
    ny : int
        Number of meshes along y.
    nz: int
        Number of meshes along z.
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
    csc_matrix
        The sparse precision matrix.
    """

    # Check if 2 alpha is an integer
    if alpha < 1.0 or not float(alpha).is_integer():
        raise ValueError(
            "alpha must be superior or equal to 1.0 and must be an whole number!"
        )
    # Discretization of (kappa^2 - Delta)^(alpha)
    # Build the laplacian matrix: (kappa^2 - Delta)
    A: csc_matrix = get_laplacian_matrix(nx, ny, nz, dx, dy, dz, kappa)

    # Apply alpha (we deal only with integers alpha)
    Af = sp.sparse.identity(A.shape[0])

    # Use mass lumping
    for i in range(int(alpha)):
        # Af = A @ Af  # matrix multiplication
        Af = A @ Af

    # Correction factor for variance
    nu = 2 * alpha - spatial_dim / 2
    tau = (kappa ** (nu)) * np.sqrt(
        (4 * np.pi) * np.math.gamma(2 * alpha) / np.math.gamma(nu)
    )

    # Calculate precision matrix
    Af = np.sqrt((dx * dy)) * Af / (tau * sigma)
    return (Af.T @ Af).tocsc()


def simu_nc(
    cholQ: Factor,
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
    if random_state is not None:
        random_state = check_random_state(random_state)
    else:
        random_state = np.random.default_rng()

    w = random_state.normal(size=cholQ.L().shape[0])  # white noise
    return cholQ.apply_Pt(cholQ.solve_Lt(1.0 / np.sqrt(cholQ.D()) * w))


def condition_precision_matrix(
    Q: csc_matrix, dat_indices: NDArrayInt, dat_var: NDArrayFloat
) -> csc_matrix:
    """
    Condition the precision matrix with the variance of known data points.

    Parameters
    ----------
    Q : csc_matrix
        _description_
    dat_indices : NDArrayInt
        _description_
    dat_var : NDArrayFloat
        _description_

    Returns
    -------
    csc_matrix
        The conditioned precision matrix.
    """
    # Build the diagonal matrix containing the inverse of the error variance at known
    # data points

    diag_var = lil_matrix(Q.shape)
    diag_var[dat_indices, dat_indices] = 1 / dat_var
    return (diag_var + Q).tocsc()


def kriging(
    Q: csc_matrix,
    dat: NDArrayFloat,
    dat_indices: NDArrayInt,
    cholQ: Optional[Factor] = None,
    dat_var: Optional[NDArrayFloat] = None,
) -> NDArrayFloat:
    if cholQ is None:
        _cholQ = cholesky(Q.tocsc())
    else:
        _cholQ = cholQ
    input = np.zeros(Q.shape[0])
    input[dat_indices] = dat
    if dat_var is not None:
        input[dat_indices] /= dat_var
    return _cholQ(input)


def simu_c(
    cholQ: Factor,
    Q_cond: csc_matrix,
    cholQ_cond: Factor,
    dat: NDArrayFloat,
    dat_indices: NDArrayInt,
    dat_var: NDArrayFloat,
    random_state: Optional[Union[int, np.random.Generator, np.random.RandomState]],
) -> NDArrayFloat:
    """_summary_

    Parameters
    ----------
    cholQ : Factor
        _description_
    Q_cond : csc_matrix
        _description_
    cholQ_cond : Factor
        _description_
    dat : NDArrayFloat
        _description_
    dat_indices : NDArrayInt
        _description_
    dat_var : NDArrayFloat
        _description_
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
        _description_
    """
    z_k = kriging(Q_cond, dat, dat_indices, cholQ=cholQ_cond, dat_var=dat_var)
    # z_k = krig_prec2(Q_cond, dat * 1 / grid_var[dat_indices], dat_indices)
    z_nc = simu_nc(cholQ, random_state)
    dat_nc = z_nc[dat_indices]
    # z_nck = krig_chol(QTT_factor, QTD, dat_nc, dat_indices)
    z_nck = kriging(Q_cond, dat_nc, dat_indices, cholQ=cholQ_cond, dat_var=dat_var)
    return z_k - (z_nc - z_nck)
