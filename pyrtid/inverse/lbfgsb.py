"""
@author: Antoine COLLET.

This code is a python port of the famous implementation of Limited-memory
Broyden-Fletcher-Goldfarb-Shanno (L-BFGS), algorithm 778 written in Fortran [2,3]
(last update in 2011).
Note that this is not a wrapper such as minimize in scipy but a complete
reimplementation (pure python).
The original code can be found here: https://dl.acm.org/doi/10.1145/279232.279236

The aim of this reimplementation was threefold. First, familiarize ourselves with
the code, its logic and inner optimizations. Second, gain access to certain
parameters that are hard-coded in the Fortran code and cannot be modified (typically
wolfe conditions parameters for the line search). Third,
implement additional functionalities that require significant modification of
the code core.

Additional features
--------------------
Explain about objective function update on the fly.
TODO: point to the doc of the main routine.

References
----------
[1] R. H. Byrd, P. Lu and J. Nocedal. A Limited Memory Algorithm for Bound
    Constrained Optimization, (1995), SIAM Journal on Scientific and
    Statistical Computing, 16, 5, pp. 1190-1208.
[2] C. Zhu, R. H. Byrd and J. Nocedal. L-BFGS-B: Algorithm 778: L-BFGS-B,
    FORTRAN routines for large scale bound constrained optimization (1997),
    ACM Transactions on Mathematical Software, 23, 4, pp. 550 - 560.
[3] J.L. Morales and J. Nocedal. L-BFGS-B: Remark on Algorithm 778: L-BFGS-B,
    FORTRAN routines for large scale bound constrained optimization (2011),
    ACM Transactions on Mathematical Software, 38, 1.
"""
import copy
from collections import deque
from enum import Enum, auto
from typing import Callable, Deque, Optional, Tuple, Union

import numpy as np
import scipy as sp
from scipy.optimize import minpack2
from scipy.optimize._constraints import old_bound_to_new
from scipy.optimize._lbfgsb_py import LbfgsInvHessProduct  # noqa : F401
from scipy.optimize._optimize import (
    OptimizeResult,
    _check_unknown_options,  # noqa : F401
    _prepare_scalar_function,
)
from scipy.sparse import lil_array, spmatrix

from pyrtid.utils import NDArrayFloat, NDArrayInt


class Status(Enum):
    """Representation of the solver status."""

    START = auto()
    ERROR = auto()


class LBFGSmat:
    """Represent the LBFGS matrices."""

    __slots__ = ["Wa", "Ws", "Wy", "Sy", "Ss", "Wn", "Wt", "Snd"]

    def __init__(self, n: int, m: int) -> None:
        """Initialize the instance.

        Parameters
        ----------
        n : int
            The number of variables in the optimization problem.
        m : int
            The mamximum number of variables metric correction in the limited
            memory matrix (number of gradients stored).
        """
        # Create the working arrays used to store information defining the limited
        # memory BFGS matrix -> fixed memory allocation

        # working vector -> this is a vector of dim (8 * m)
        # This vector is used to store the intermediate calculation when finding
        # the cauchy points
        self.Wa = np.zeros((8 * m), dtype=np.float_)
        # Ws stores S, the matrix of s-vectors;
        self.Ws = np.zeros((n, m), dtype=np.float_)
        # Wy stores Y, the matrix of y-vectors;
        self.Wy = np.zeros((n, m), dtype=np.float_)
        # Sy stores S'Y;
        self.Sy = np.zeros((m, m), dtype=np.float_)
        # Ss stores S'S;
        self.Ss = np.array((m, m), dtype=np.float_)
        # Wn stores the Cholesky factorization of (theta*S'S+LD^(-1)L');
        # see eq. (2.26) in [3].
        self.Wn = np.zeros((2 * m, 2 * m), dtype=np.float_)
        # used to store the LEL^T factorization of the indefinite matrix
        # ```
        #  K = [-D -Y'ZZ'Y/theta     L_a'-R_z'  ]
        #      [L_a -R_z           theta*S'AA'S ]
        # ```
        # where
        # ```
        #  E = [-I  0]
        #      [ 0  I]
        # ```
        self.Wt = np.zeros((m, m), dtype=np.float_)
        # stores the lower triangular part of
        # ```
        #  N = [Y' ZZ'Y   L_a'+R_z']
        #      [L_a +R_z  S'AA'S   ]
        # ```
        self.Snd = np.zeros((2 * m, 2 * m), dtype=np.float_)


def get_cauchy_point(
    x: NDArrayFloat,
    grad: NDArrayFloat,
    lb: NDArrayFloat,
    ub: NDArrayFloat,
    W: NDArrayFloat,
    M: NDArrayFloat,
    invMlt: NDArrayFloat,
    mats: LBFGSmat,
    theta: float,
    col: int,
    max_cor: int,
    iprint: int,
    iter: int,
):
    r"""
    Computes the generalized Cauchy point (GCP).

    This is the Generalized Cauchy point procedure in section 4 of [1].

    It is defined as the first local minimizer of the quadratic

    .. math::
        \[\langle grad,s\rangle + \frac{1}{2} \langle s,
        (\theta I + WMW^\intercal)s\rangle\]

    along the projected gradient direction .. math::`P_[l,u](x-\theta grad).`

    Parameters
    ----------
    x : NDArrayFloat
        Starting point for the GCP computation.
    grad : NDArrayFloat
        Gradient of fun with respect to x.
    l : NDArrayFloat
        Lower bound vector.
    u : NDArrayFloat
        Upper bound vector.
    W : NDArrayFloat
        Part of limited memory BFGS Hessian approximation
    M : NDArrayFloat
        Part of limited memory BFGS Hessian approximation
    mats: LBFGSmat
        LBFGS matrices.
    theta : float
        Part of limited memory BFGS Hessian approximation.
    col: int
        The actual number of variable metric corrections stored so far.
    iprint: int
        Printing level.
    iter: int
        Current iteration.

    Returns
    -------
    Dict
        Dict containing a computed value of:
        - 'xc' the GCP
        - 'c' = W^(T)(xc-x), used for the subspace minimization
        - 'F' set of free variables

    References
    ----------
    * R. H. Byrd, P. Lu and J. Nocedal. A Limited Memory Algorithm for Bound
      Constrained Optimization, (1995), SIAM Journal on Scientific and
      Statistical Computing, 16, 5, pp. 1190-1208.
    * C. Zhu, R. H. Byrd and J. Nocedal. L-BFGS-B: Algorithm 778: L-BFGS-B,
      FORTRAN routines for large scale bound constrained optimization (1997),
      ACM Transactions on Mathematical Software, 23, 4, pp. 550 - 560.
    * J.L. Morales and J. Nocedal. L-BFGS-B: Remark on Algorithm 778: L-BFGS-B,
      FORTRAN routines for large scale bound constrained optimization (2011),
      ACM Transactions on Mathematical Software, 38, 1.
    """
    if iprint >= 99:
        print("---------------- CAUCHY entered-------------------")

    eps_f_sec = 1e-30
    x_cp = x.copy()

    x.size
    m = mats.Ws.shape[1]

    # These 4 vectors are Wa (8 * m)
    # working array used to store the vector `p = W^(T)d`.
    p = mats.Wa[: 2 * m]
    # working array used to store the vector `c = W^(T)(xcp-x)`.
    c = mats.Wa[2 * m + 1 : 4 * m]
    # working array used to store the row of `W` corresponding to a breakpoint.
    mats.Wa[4 * m + 1 : 6 * m]
    # working array
    mats.Wa[6 * m + 1 :]

    # To define the breakpoints in each coordinate direction, we compute
    t = np.where(grad < 0, (x - ub) / grad, (x - lb) / grad)
    t[grad == 0] = np.inf

    # used to store the Cauchy direction `P(x-tg)-x`.
    d = np.where(t == 0, 0.0, -grad)

    # sort {t;,i = 1,. ..,n} in increasing order to obtain the ordered
    # set {tj :tj <= tj+1 ,j = 1, ...,n}.
    # Keep only the indices where t > 0
    F = np.argsort(t)[t > 0]

    # F = [i for i in F if t[i] > 0]
    # In the end, F is the list of ordered breakpoint indices

    # TODO: handle the case with no breakpoints
    # if ( nbreak==0 .and. nfree==n+1 ) then ...

    # TODO: The integer t denotes the number of free variables at the Cauchy point zc;
    # in other words there are n - t variables at bound at zC

    # Initialization
    # There is a problem with the size of W -> it should be fixed but it is not here....
    # See what is best with python ???
    # TODO: p[:] = W.T @ d
    p = W.T @ d  # 2mn operations
    # TODO: c[:] = 0
    c = np.zeros(p.size)
    # f1 in the original code
    f_prime: float = -d.dot(d)  # n operations
    # f2 in the original code
    f_second: float = -theta * f_prime
    # f2_org in the fortran code
    f_sec0: float = copy.deepcopy(f_second)
    # Update f2 with - d^{T} @ W @ M @ W^{T} @ d = - p^{T} @ M @ p
    # old way: f_second = f_second - p.dot(M.dot(p))  # O(m^{2}) operations
    # new_way: not at first iteration -> invMlt and M are worse zero.
    # And cho_solve produces nan
    if iter != 0:
        f_second = f_second - p.dot(
            sp.linalg.cho_solve((invMlt, True), p)
        )  # O(m^{2}) operations

    # dtm in the fortran code
    Dt_min: float = -f_prime / f_second

    # iter in the fortran code
    F_i = 0
    # break point index (b in section 4 [1])
    ibp = F[F_i]  # TODO: remove b from F ???
    # value of the smallest breakpoint, t in section 4 [1]
    t_min = t[ibp]
    # previous breakpoint value
    t_old = 0

    Dt = t_min - 0

    # Number of the breakpoint segment -> Nseg in Fortran
    nseg: int = 1  # TODO: check that

    nbreak = len(F)
    if iprint >= 99:
        print(f"There are {nbreak} breakpoints ")

    if nbreak != 0:
        pass

        while Dt_min >= Dt and F_i < len(F):
            if Dt != 0 and iprint >= 100:
                print(
                    f"Piece    , {nseg},  --f1, f2 at start point , {f_prime} , "
                    f"{f_second}"
                )
                print(f"Distance to the next break point =  {Dt}")
                print(f"Distance to the stationary point =  {Dt_min}")

            # Fix one variable and reset the corresponding component of d to zero.
            if d[ibp] > 0:
                x_cp[ibp] = ub[ibp]
            elif d[ibp] < 0:
                x_cp[ibp] = lb[ibp]
            x_bcp = x_cp[ibp]
            zb = x_bcp - x[ibp]

            if iprint >= 100:
                # ibp +1 to match the Fortran code (because index starts at 1)
                print(f"Variable  {ibp + 1} is fixed.")
            F_i += 1

            c += Dt * p
            W_b = W[ibp, :]
            g_b = grad[ibp]

            # Update the derivative information
            # 1) Old way
            # f_prime += Dt * f_second + g_b * (g_b + theta * zb - W_b.dot(M.dot(c)))
            # f_second -= g_b * (g_b * theta + W_b.dot(M.dot(2 * p + g_b * W_b)))
            # 2) New way with the cholesky factorization
            f_prime += Dt * f_second + g_b * (g_b + theta * zb)
            f_second -= g_b * (g_b * theta)
            # First iteration -> invMlt and M are worse zero.
            # And cho_solve produces nan
            if iter != 0:
                f_prime -= g_b * W_b.dot(sp.linalg.cho_solve((invMlt, True), c))
                f_second -= g_b * W_b.dot(
                    sp.linalg.cho_solve((invMlt, True), (2 * p + g_b * W_b))
                )

            f_second = min(f_second, eps_f_sec * f_sec0)

            Dt_min = -f_prime / f_second

            # Fix one variable and reset the corresponding component of d to zero.
            p += g_b * W_b
            d[ibp] = 0
            t_old = t_min

            if F_i < len(F):
                ibp = F[F_i]
                t_min = t[ibp]
                Dt = t_min - t_old
            else:
                t_min = np.inf

            nseg += 1

    if iprint >= 99:
        print("GCP found in this segment")

        # print(f"Piece    {nseg}  --f1, f2 at start point , {f_prime} , {f_second}")
        # print(f"Distance to the stationary point = {Dt}")

    Dt_min = 0 if Dt_min < 0 else Dt_min
    t_old += Dt_min

    x_cp[t >= t_min] = (x + t_old * d)[t >= t_min]

    F = [i for i in F if t[i] != t_min]

    c += Dt_min * p

    if iprint > 100:
        print(f"Cauchy X =  {x_cp}")
    if iprint >= 99:
        print("---------------- exit CAUCHY----------------------")

    return {
        "xc": x_cp,
        "c": c,
        "F": F,
    }


def freev(
    x_cp: NDArrayFloat,
    lb: NDArrayFloat,
    ub: NDArrayFloat,
    free_vars_old: NDArrayInt,
    iprint: int,
    iter: int,
) -> Tuple[NDArrayInt, spmatrix, spmatrix]:
    """
    Get the free variables and build Z and A matrices (sparse).

    Parameters
    ----------
    x_cp : NDArrayFloat
        Generalized cauchy point.
    lb : NDArrayFloat
        Lower bounds.
    ub : NDArrayFloat
        Upper bounds.
    free_vars_old : NDArrayInt
        Free variables at x_cp at the previous iteration.
    iprint : int
        Level of display.
    iter : int
        Iteration number.

    Returns
    -------
    Tuple[NDArrayInt, spmatrix, spmatrix]
        The free variables and sparse matrices Z and A.
    """
    # number of variables
    n: int = x_cp.size

    # Array of free variable and active variable indices (from 0 to n-1)
    free_vars: NDArrayInt = ((x_cp != ub) & (x_cp != lb)).nonzero()[0]
    active_vars: NDArrayInt = (
        ~np.isin(np.arange(n), free_vars)  # type: ignore
    ).nonzero()[0]

    nb_free_vars: int = free_vars.size
    nb_active_vars: int = active_vars.size

    # See section 5 of [1]: We define Z to be the (n , t) matrix whose columns are
    # unit vectors (i.e., columns of the identity matrix) that span the subspace of the
    # free variables at zc.Similarly A denotes the (n, (n- t)) matrix of active
    # constraint gradients at zc,which consists of n - t unit vectors.
    # Note that A^{T}Z = 0 and that  AA^T + ZZ^T == I.

    # We use sparse formats to save memory and get faster matrix products
    Z = lil_array((n, nb_free_vars))
    A = lil_array((n, nb_active_vars))
    # Affect one
    Z[free_vars, np.arange(nb_free_vars)] = 1
    A[active_vars, np.arange(nb_active_vars)] = 1

    # Test: we should have Z @ Z.T + A @ A.T == I

    # Some display
    # 1) Indicate which variable is leaving the free variables and which is
    # entering the free variables -> Not for the first iteration
    if iprint > 100 and iter > 0:
        # Variables leaving the free variables
        leaving_vars = active_vars[np.isin(active_vars, free_vars_old)]
        print(f"Variables leaving the free variables set = {leaving_vars}")
        entering_vars = free_vars[~np.isin(free_vars, free_vars_old)]
        print(f"Variables entering the free variables set = {entering_vars}")
        print(
            f"N variables leaving = {leaving_vars.size} \t,"
            f" N variables entering = {entering_vars.size}"
        )
    # 2) Display the total of free variables at x_cp
    if iprint > 99:
        print(f"{free_vars.size} variables are free at GCP, iter = {iter + 1}")

    return free_vars, Z.tocsc(), A.tocsc()


# There are three methods for this one and we need to find the correct one.
def direct_primal_subspace_minimization(
    x: NDArrayFloat,
    xc: NDArrayFloat,
    free_vars: NDArrayInt,
    Z: spmatrix,
    c: NDArrayFloat,
    grad: NDArrayFloat,
    lb: NDArrayFloat,
    ub: NDArrayFloat,
    W: NDArrayFloat,
    M: NDArrayFloat,
    theta: float,
    K: NDArrayFloat,
) -> NDArrayFloat:
    r"""
    Computes an approximate solution of the subspace problem.

    This is following section 5.1 in Byrd et al. (1995).

    .. math::
        :nowrap:

       \[\begin{aligned}
            \min& &\langle r, (x-xcp)\rangle + 1/2 \langle x-xcp, B (x-xcp)\rangle\\
            \text{s.t.}& &l<=x<=u\\
                       & & x_i=xcp_i \text{for all} i \in A(xcp)
        \]

    along the subspace unconstrained Newton direction
    .. math:: $d = -(Z'BZ)^(-1) r.$


    # TODO Normally, free_vars is already defined in compute_Cauchy_point in F.

    Parameters
    ----------
    x : NDArrayFloat
        Starting point for the GCP computation
    xc : NDArrayFloat
        Cauchy point.
    c : NDArrayFloat
        W^T(xc-x), computed with the Cauchy point.
    grad : NDArrayFloat
        Gradient of f(x). grad must be a nonzero vector.
    lb : NDArrayFloat
        Lower bound vector.
    ub : NDArrayFloat
        Upper bound vector.
    W : NDArrayFloat
        Part of limited memory BFGS Hessian approximation.
    M : NDArrayFloat
        Part of limited memory BFGS Hessian approximation.
    theta : float
        Part of limited memory BFGS Hessian approximation.
    Z: spmatrix
        Warning: it has shape (n, t)

    Returns
    -------
    NDArrayFloat
        xbar

    References
    ----------
    * R. H. Byrd, P. Lu and J. Nocedal. A Limited Memory Algorithm for Bound
      Constrained Optimization, (1995), SIAM Journal on Scientific and
      Statistical Computing, 16, 5, pp. 1190-1208.
    * C. Zhu, R. H. Byrd and J. Nocedal. L-BFGS-B: Algorithm 778: L-BFGS-B,
      FORTRAN routines for large scale bound constrained optimization (1997),
      ACM Transactions on Mathematical Software, 23, 4, pp. 550 - 560.
    * J.L. Morales and J. Nocedal. L-BFGS-B: Remark on Algorithm 778: L-BFGS-B,
      FORTRAN routines for large scale bound constrained optimization (2011),
      ACM Transactions on Mathematical Software, 38, 1.
    """
    # Direct primal method

    invThet = 1.0 / theta

    if len(free_vars) == 0:
        return xc

    # Same as W.T.dot(Z) but numpy does not handle correctly
    # numpy_array.dot(sparce_matrix), so we give the responsibility to the
    # sparse matrix
    # Note that here, Z is suppose to have a shape (t, n) with t the number
    # of free_vars and n the number of variables.
    # WTZ = W.T.dot(Z.todense())
    WTZ = Z.T.dot(W).T

    rHat = [(grad + theta * (xc - x) - W.dot(M.dot(c)))[ind] for ind in free_vars]
    v = M.dot(WTZ.dot(rHat))

    N = -M.dot(invThet * WTZ.dot(np.transpose(WTZ)))
    # N = invThet * WTZ.dot(np.transpose(WTZ))
    # Add the identitu matrix: this is the same as N = np.eye(N.shape[0]) - M.dot(N)
    # but much faster
    np.fill_diagonal(N, N.diagonal() + 1)

    # TODO: this is not efficient at all and we should try to remove it
    v = np.linalg.solve(N, v)

    # TODO: new way to perform
    # v = sp.linalg.solve(K, v)

    dHat = -invThet * (rHat + invThet * np.transpose(WTZ).dot(v))

    # Find alpha
    # TODO: remove the loop
    alpha_star = 1
    for i in range(len(free_vars)):
        idx = free_vars[i]
        if dHat[i] > 0:
            alpha_star = min(alpha_star, (ub[idx] - xc[idx]) / dHat[i])
        elif dHat[i] < 0:
            alpha_star = min(alpha_star, (lb[idx] - xc[idx]) / dHat[i])

    d_star = alpha_star * dHat
    xbar = xc
    for i in range(len(free_vars)):
        idx = free_vars[i]
        xbar[idx] += d_star[i]

    return xbar


def formk(X: Deque, G: Deque, Z: spmatrix, A: spmatrix, theta: float) -> NDArrayFloat:
    """Form mk

    Form  the LEL^T factorization of the indefinite
    matrix    K = [-D -Y'ZZ'Y/theta     L_a'-R_z'  ]
                    [L_a -R_z           theta*S'AA'S ]
    where     E = [-I  0]
                    [ 0  I]

    TODO

    Parameters
    ----------
    """
    # form S and Y
    S = np.diff(np.array(X), axis=0).T
    Y = np.diff(np.array(G), axis=0).T
    D: NDArrayFloat = np.diag(np.diag(S.T @ Y))

    # K sub-blocks

    # LZ is the upper triangular part
    if Z.size == 0:
        YTZZTY = np.array([0.0])
        LZ = np.array([0.0])
    else:
        YTZZTY = Y.T @ Z @ Z.T @ Y
        LZ = np.triu(YTZZTY)

    # LA is the strict lower triangle of S^{T}AA^{T}S
    if A.size == 0:
        STAATS = np.array([0.0])
        LA = np.array([0.0])
    else:
        STAATS = S.T @ A @ A.T @ S
        LA = np.tril(STAATS, -1)

    K11: NDArrayFloat = -D - 1 / theta * YTZZTY
    K21: NDArrayFloat = LA - LZ

    if A.size == 0:
        K22 = np.zeros(K21.shape)
    else:
        K22 = theta * STAATS

    try:
        print(K11)
        L11 = sp.linalg.cholesky(K11, lower=True)
        print(L11)
        S = K22 - K21 @ sp.linalg.cho_solve((L11, True), K21.T)
    except ValueError:
        S = 0.0
    # print(K11.shape)
    # print(K21.shape)
    # print(K22.shape)

    return np.hstack([np.vstack([K11, K21]), np.vstack([K21.T, K22])])


def max_allowed_steplength(
    x: NDArrayFloat,
    d: NDArrayFloat,
    lb: NDArrayFloat,
    ub: NDArrayFloat,
    max_steplength: float,
) -> float:
    r"""
    Computes the biggest 0<=k<=max_steplength such that:
        l<= x+kd <= u

    Parameters
    ----------
    x : NDArrayFloat
        Starting point.
    d : NDArrayFloat
        Direction.
    lb : NDArrayFloat
        the lower bound of x.
    ub : NDArrayFloat
        The upper bound of x
    max_steplength : float
        Maximum steplength allowed.

    Returns
    -------
    float
        maximum steplength allowed

    References
    ----------
    * R. H. Byrd, P. Lu and J. Nocedal. A Limited Memory Algorithm for Bound
      Constrained Optimization, (1995), SIAM Journal on Scientific and
      Statistical Computing, 16, 5, pp. 1190-1208.
    * C. Zhu, R. H. Byrd and J. Nocedal. L-BFGS-B: Algorithm 778: L-BFGS-B,
      FORTRAN routines for large scale bound constrained optimization (1997),
      ACM Transactions on Mathematical Software, 23, 4, pp. 550 - 560.
    * J.L. Morales and J. Nocedal. L-BFGS-B: Remark on Algorithm 778: L-BFGS-B,
      FORTRAN routines for large scale bound constrained optimization (2011),
      ACM Transactions on Mathematical Software, 38, 1.
    """
    with np.errstate(divide="ignore"):
        _tmp = np.where(d > 0, (ub - x) / d, (lb - x) / d)
        return min(max_steplength, np.nanmin(_tmp[np.isfinite(_tmp)]))


def line_search(
    x0: NDArrayFloat,
    f0: float,
    g0: NDArrayFloat,
    d: NDArrayFloat,
    above_iter: int,
    max_steplength: float,
    fun_and_grad: Callable[[NDArrayFloat], Tuple[float, NDArrayFloat]],
    ftol: float = 1e-4,  # called ftol and = 1e-3 in algo 778
    gtol: float = 0.9,  # called gtol and = 0.9 in algo 778
    xtol: float = 1e-5,  # called xtol = 0.1 in alga 778
    max_iter: int = 30,
    iprint: int = 10,
) -> Optional[float]:
    r"""
    Find a step that satisfies both decrease condition and a curvature condition.

        f(x0+stp*d) <= f(x0) + alpha*stp*\langle f'(x0),d\rangle,

    and the curvature condition

        abs(f'(x0+stp*d)) <= beta*abs(\langle f'(x0),d\rangle).

    If alpha is less than beta and if, for example, the functionis bounded below, then
    there is always a step which satisfies both conditions.

    Note
    ----
    This subroutine calls subroutine dcsrch from the Minpack2 library
    to perform the line search.  Subroutine dscrch is safeguarded so
    that all trial points lie within the feasible region.

    Parameters
    ----------
    x0 : NDArrayFloat
        Starting point.
    f0 : float
        Objective function value for x0.
    g0 : NDArrayFloat
        Gradient of the objective function for x0.
    d : NDArrayFloat
        Search direction.
    above_iter : int
        current iteration in optimization process.
    max_steplength : float
        Maximum steplength allowed.
    fun_and_grad : Callable[[NDArrayFloat], Tuple[float, NDArrayFloat]]
        Function returning both the obejctive function and its gradient with respect to
        a given vector x.
    ftol_linesearch: float, optional
        Specify a nonnegative tolerance for the sufficient decrease condition in
        `minpack2.dcsrch <https://ftp.mcs.anl.gov/pub/MINPACK-2/csrch/dcsrch.f>`_
        (used for the line search). This is :math:`c_1` in
        the Armijo condition (or Goldstein, Goldstein-Armijo condition) where
        :math:`\alpha_{k}` is the estimated step.

        .. math::

            f(\mathbf{x}_{k}+\alpha_{k}\mathbf{p}_{k})\leq
            f(\mathbf{x}_{k})+c_{1}\alpha_{k}\mathbf{p}_{k}^{\mathrm{T}}
            \nabla f(\mathbf{x}_{k})

        Note that :math:`0 < c_1 < 1`. Usually :math:`c_1` is small, see the Wolfe
        conditions in :cite:t:`nocedalNumericalOptimization1999`.
        In the fortran implementation
        algo 778, it is hardcoded to 1e-3. The default is 1e-4.
    gtol_linesearch: float, optional
        Specify a nonnegative tolerance for the curvature condition in
        `minpack2.dcsrch <https://ftp.mcs.anl.gov/pub/MINPACK-2/csrch/dcsrch.f>`_
        (used for the line search). This is :math:`c_2` in
        the Armijo condition (or Goldstein, Goldstein-Armijo condition) where
        :math:`\alpha_{k}` is the estimated step.

        .. math::

            \left|\mathbf{p}_{k}^{\mathrm {T}}\nabla f(\mathbf{x}_{k}+\alpha_{k}
            \mathbf{p}_{k})\right|\leq c_{2}\left|\mathbf {p}_{k}^{\mathrm{T}}\nabla
            f(\mathbf{x}_{k})\right|

        Note that :math:`0 < c_1 < c_2 < 1`. Usually, :math:`c_2` is
        much larger than :math:`c_2`.
        see :cite:t:`nocedalNumericalOptimization1999`. In the fortran implementation
        algo 778, it is hardcoded to 0.9. The default is 0.9.
    xtol_linesearch: float, optional
        Specify a nonnegative relative tolerance for an acceptable step in the line
        search procedure (see
        `minpack2.dcsrch <https://ftp.mcs.anl.gov/pub/MINPACK-2/csrch/dcsrch.f>`_).
        In the fortran implementation algo 778, it is hardcoded to 0.1.
        The default is 1e-5.
    max_iter : int, optional
        Maximum number of linesearch iterations, by default 30.

    Returns
    -------
    Optional[float]
        The step length.

    References
    ----------
    * R. H. Byrd, P. Lu and J. Nocedal. A Limited Memory Algorithm for Bound
      Constrained Optimization, (1995), SIAM Journal on Scientific and
      Statistical Computing, 16, 5, pp. 1190-1208.
    * C. Zhu, R. H. Byrd and J. Nocedal. L-BFGS-B: Algorithm 778: L-BFGS-B,
      FORTRAN routines for large scale bound constrained optimization (1997),
      ACM Transactions on Mathematical Software, 23, 4, pp. 550 - 560.
    * J.L. Morales and J. Nocedal. L-BFGS-B: Remark on Algorithm 778: L-BFGS-B,
      FORTRAN routines for large scale bound constrained optimization (2011),
      ACM Transactions on Mathematical Software, 38, 1.
    """

    # steplength_0 = 1 if max_steplength > 1 else 0.5 * max_steplength

    f_m1 = f0
    dphi = g0.dot(d)
    dphi_m1 = dphi
    i = 0

    if above_iter == 0:
        steplength_0 = min(1.0 / np.sqrt(d.dot(d)), max_steplength)
    else:
        steplength_0 = 1.0

    # print(f"max_steplength = {max_steplength}")
    # print(f"steplength_0 = {steplength_0}")

    isave = np.zeros((2,), np.intc)
    dsave = np.zeros((13,), float)
    task = b"START"

    while i < max_iter:
        steplength, f0, dphi, task = minpack2.dcsrch(
            steplength_0,
            f_m1,
            dphi_m1,
            ftol,
            gtol,
            xtol,
            task,
            0,
            max_steplength,
            isave,
            dsave,
        )
        if task[:2] == b"FG":
            steplength_0 = steplength
            f_m1, dphi_m1 = fun_and_grad(x0 + steplength * d)
            dphi_m1 = dphi_m1.dot(d)
        else:
            break
    else:
        # max_iter reached, the line search did not converge
        steplength = None

    if task[:5] == b"ERROR" or task[:4] == b"WARN":
        if task[:21] != b"WARNING: STP = STPMAX":
            print(task)
            steplength = None  # failed

    if iprint >= 99:
        print(f"LINE SEARCH  {i} times; norm of step = {steplength}")

    return steplength


def get_lbfgs_matrices(
    xk: NDArrayFloat,
    gk: NDArrayFloat,
    X: Deque[NDArrayFloat],
    G: Deque[NDArrayFloat],
    maxcor: int,
    mats: LBFGSmat,
    W: NDArrayFloat,
    M: NDArrayFloat,
    invMlt: NDArrayFloat,
    theta: float,
    is_force_update: bool,
    eps: float = 2.2e-16,
) -> Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, float]:
    r"""
    Update lists S and Y, and form the L-BFGS Hessian approximation thet, W and M.

    Instead of storing sk and yk, we store the gradients and the parameters.

    2 conditions for update
    - The current step update is accepted
    - The all sequence of x and g has been modified (reg case)

    Parameters
    ----------
    xk : NDArrayFloat
        New x parameter.
    gk : NDArrayFloat
        New gradient parameter g.
    X : deque
        List of successive parameters x.
    G : deque
        List of successive gradients.
    maxcor : int
        The maximum number of variable metric corrections used to
        define the limited memory matrix. (The limited memory BFGS
        method does not store the full hessian but uses this many terms
        in an approximation to it.)
    W : NDArrayFloat
        L-BFGS matrices.
    M : NDArrayFloat
        L-BFGS matrices.
    thet : float
        L-BFGS float parameter (multiply the identity matrix).
    is_force_update: bool
        Whether to perform an update even if the current step update is rejected.
        This is useful if the sequence of X and G has been modified during the
        optimization. See TODO: add ref, for the use.
    eps : float, optional
        Positive stability parameter for accepting current step for updating.
        By default 2.2e-16.

    Returns
    -------
    List[NDArrayFloat, NDArrayFloat, float]
        Updated [W, M, thet]

    References
    ----------
    * R. H. Byrd, P. Lu and J. Nocedal. A Limited Memory Algorithm for Bound
      Constrained Optimization, (1995), SIAM Journal on Scientific and
      Statistical Computing, 16, 5, pp. 1190-1208.
    * C. Zhu, R. H. Byrd and J. Nocedal. L-BFGS-B: Algorithm 778: L-BFGS-B,
      FORTRAN routines for large scale bound constrained optimization (1997),
      ACM Transactions on Mathematical Software, 23, 4, pp. 550 - 560.
    * J.L. Morales and J. Nocedal. L-BFGS-B: Remark on Algorithm 778: L-BFGS-B,
      FORTRAN routines for large scale bound constrained optimization (2011),
      ACM Transactions on Mathematical Software, 38, 1.
    """
    yk = gk - G[-1]
    sTy = (xk - X[-1]).dot(yk)  # type: ignore
    yTy = (yk).dot(yk)  # type: ignore

    is_current_update_accepted = False

    # TODO: Why do we perform that check ? # See eq. (3.9) in [1].
    if sTy > eps * yTy:
        is_current_update_accepted = True
        X.append(xk)
        G.append(gk)
        if len(X) > maxcor:
            X.popleft()
            G.popleft()

    theta = 1.0

    # two conditions to update the inverse Hessian approximation
    if is_force_update or is_current_update_accepted:
        # Update the lbfgsb matrices
        Sarray = np.diff(np.array(X), axis=0).T  # shape (n, m - 1)
        Yarray = np.diff(np.array(G), axis=0).T  # shape (n ,m - 1)
        STS = np.transpose(Sarray).dot(Sarray)
        L = np.transpose(Sarray).dot(Yarray)
        # We can build a dense matrix because shape is (m, m) with m usually small ~10
        D = np.diag(np.diag(L))
        L = np.tril(L, -1)

        theta = yTy / sTy
        W = np.hstack([Yarray, theta * Sarray])

        # To avoid forming the limited-memory iteration matrix Bk and allow fast
        # matrix vector products, we represent it as eq. (3.2) [1].
        # B = theta * I  - W @ M @ W.T

        # M (or Mk) can be obtained with
        M = np.linalg.inv(
            np.hstack([np.vstack([-D, L]), np.vstack([L.T, theta * STS])])
        )
        # However, we can also factorize its inverse and obtain very fast matrix
        # products: lower triangle of M inverse
        invMlt = form_invMlt(theta, STS, L, D)

    return W, M, invMlt, theta


def form_invMlt(theta, STS, L, D) -> NDArrayFloat:
    r"""
    Perform the cholesky factorization of the inverse of M_k, defined in eq. (3.4) [1].

    Although Mk is not positive definite, but its inverse reads:

        [  -D       L'        ]
        [   L       theta * S'*S]

    Hence its inverse can be factorized symmetrically by using Cholesky factorizations
    of the submatrices TODO: add ref to the phd manuscript.
    Now, the inverse of Mk, the middle matrix in B reads:

         [  D^(1/2)      O ] [ -D^(1/2)  D^(-1/2)*L' ]
         [ -L*D^(-1/2)   J ] [  0        J'          ]

    With J*J' = T = theta*Ss + L*D^(-1)*L'; T being definite positive, J is obtained by
    Cholesky factorization of T.
    """
    # Cholesky factorization
    J = sp.linalg.cholesky(theta * STS + L @ sp.linalg.solve(D, L.T), lower=True)

    # Note we form the upper triangle and then transpose it to get the lower one
    return np.hstack(
        [
            np.vstack([-np.sqrt(D), sp.linalg.solve(np.sqrt(D), L.T)]),  # upper row
            np.vstack([np.zeros(D.shape), J.T]),  # lower row
        ]
    ).T


def get_bounds(
    x0: NDArrayFloat, bounds: Optional[NDArrayFloat]
) -> Tuple[NDArrayFloat, NDArrayFloat]:
    if bounds is None:
        bounds = np.repeat(np.array([(-np.inf, np.inf)]), x0.size, axis=1)
    else:
        if len(bounds) != x0.size:
            raise ValueError("length of x0 != length of bounds")

    lb, ub = old_bound_to_new(bounds)

    # check bounds
    if (lb > ub).any():
        raise ValueError(
            "LBFGSB - one of the lower bounds is greater than an upper bound."
        )

    # initial vector must lie within the bounds. Otherwise ScalarFunction and
    # approx_derivative will cause problems
    return lb, ub


def clip2bounds(x0: NDArrayFloat, lb: NDArrayFloat, ub: NDArrayFloat) -> NDArrayFloat:
    """
    Impose the bounds to x0.

    Parameters
    ----------
    x0 : NDArrayFloat
        Adjusted variables.
    lb : NDArrayFloat
        Lower bounds.
    ub : NDArrayFloat
        Upper bounds.

    Returns
    -------
    NDArrayFloat
        Bounded adjusted variables.
    """
    if x0.dtype != np.float64:
        return np.clip(x0.astype(np.float64, copy=True), lb, ub)
    return np.clip(x0, lb, ub)


def count_var_at_bounds(x: NDArrayFloat, lb: NDArrayFloat, ub: NDArrayFloat) -> int:
    """
    Count the number of variables exactly at the bounds.

    Parameters
    ----------
    x : NDArrayFloat
        Adjusted variables.
    lb : NDArrayFloat
        Lower bounds.
    ub : NDArrayFloat
        Upper bounds.

    Returns
    -------
    int
        Number of variables exactly at the bounds.
    """
    return (x[x == ub]).size + (x[x == lb]).size


def display_start(epsmch, n: int, m: int, nvar_at_b: int, iprint: int) -> None:
    """
    Display information at solver start.

    Parameters
    ----------
    epsmch : _type_
        Machine precision.
    n : int
        Number of variables.
    m : int
        Number of updates.
    nvar_at_b : int
        Number of variables at bounds.
    """
    if iprint < 0:
        return
    print("RUNNING THE L-BFGS-B CODE")
    print("           * * *")
    print(f"Machine precision = {epsmch}")
    print(f"N = \t{n}\tM = \t{m}")
    print(f"At X0, {nvar_at_b} variables are exactly at the bounds")


def projgr(
    x: NDArrayFloat, grad: NDArrayFloat, lb: NDArrayFloat, ub: NDArrayFloat
) -> float:
    """
    Computes the infinity norm of the projected gradient.

    Parameters
    ----------
    x : NDArrayFloat
        _description_
    g : NDArrayFloat
        _description_
    lb : NDArrayFloat
        _description_
    ub : NDArrayFloat
        _description_

    Returns
    -------
    NDArrayFloat
        Infinity norm of the projected gradient
    """
    return np.max(np.abs(np.clip(x - grad, lb, ub) - x))


def display_iter(iter: int, sbgnrm: float, f: float, iprint: int) -> None:
    """
    Compute the infinity norm of the (-) projected gradient.

    Parameters
    ----------
    iter: int
        Current iteration number (0 to n).
    sbgnrm: float
        Infinity norm of the (-) projected gradient.
    iter: int
        Current iteration.
    iprint: int
        Level of display.
    """
    if iprint > 1:
        print(f"At iterate {iter} , f= {f} , |proj g|= {sbgnrm}")


def display_results(
    iprint: int,
    n_iterations: int,
    max_iter,
    x: NDArrayFloat,
    grad: NDArrayFloat,
    lb: NDArrayFloat,
    ub: NDArrayFloat,
    f0: float,
    gtol: float,
    is_final_display: bool,
) -> None:
    r"""
    Disaply the optimization results on the fly.

    Parameters
    ----------
    iprint : int, optional
        Controls the frequency of output. ``iprint < 0`` means no output;
        ``iprint = 0``    print only one line at the last iteration;
        ``0 < iprint < 99`` print also f and ``|proj g|`` every iprint iterations;
        ``iprint >= 99``   print details of every iteration except n-vectors;
    n_iterations : int
        _description_
    max_iter : _type_
        _description_
    x : NDArrayFloat
        _description_
    grad : NDArrayFloat
        _description_
    lb : NDArrayFloat
        Lower bound vector.
    ub : NDArrayFloat
        Upper bound vector.
    f0 : NDArrayFloat
        Last objective function value.
    gtol : float
        Relative tolerance on gradient.
    is_final_display: bool
        Is it the final display, after convergence or stop.
    """
    if iprint is None:
        return
    if iprint < 0:
        return
    if iprint == 0 and not is_final_display:
        return
    if iprint < 99 and n_iterations % iprint != 0:
        return
    print(
        "Iteration #%d (max: %d): ||x||=%.3e, f(x)=%.3e, ||jac(x)||=%.3e, "
        "cdt_arret=%.3e (eps=%.3e)"
        % (
            n_iterations,
            max_iter,
            np.linalg.norm(x, np.inf),
            f0,
            np.linalg.norm(grad, np.inf),
            projgr(x, grad, lb, ub),
            gtol,
        )
    )


def minimize_lbfgsb(
    *,
    x0: NDArrayFloat,
    fun: Callable[[NDArrayFloat, ...], float],
    args: Tuple = (),
    jac: Optional[Union[Callable[[NDArrayFloat, ...], NDArrayFloat], str, bool]],
    update_fun_def: Optional[
        Callable[
            [
                NDArrayFloat,
                float,
                NDArrayFloat,
                Deque[NDArrayFloat],
                Deque[NDArrayFloat],
            ],
            Tuple[float, NDArrayFloat, Deque[NDArrayFloat]],
        ]
    ] = None,
    bounds: Optional[NDArrayFloat] = None,
    maxcor: int = 10,
    gtol: float = 1e-5,
    ftol: float = 1e-5,
    max_iter: int = 50,
    eps: float = 1e-8,
    maxfun: int = 15000,
    iprint: int = -1,
    callback: Optional[Callable] = None,
    maxls: int = 20,
    finite_diff_rel_step: Optional[float] = None,
    max_steplength: float = 1e8,
    ftol_linesearch: float = 1e-4,
    gtol_linesearch: float = 0.9,
    xtol_linesearch: float = 1e-5,
    eps_SY: float = 2.2e-16,
) -> OptimizeResult:
    r"""
    Solves bound constrained optimization problems by using the compact formula
    of the limited memory BFGS updates.

    # TODO: try to reproduce the exact behavior of scipy.minimize

    fun :  Callable[[NDArrayFloat, Tuple[Any]], float],
        The objective function to be minimized.

            ``fun(x, *args) -> float``

        where ``x`` is a 1-D array with shape (n,) and ``args``
        is a tuple of the fixed parameters needed to completely
        specify the function.
    x0 : ndarray, shape (n,)
        Initial guess. Array of real elements of size (n,),
        where ``n`` is the number of independent variables.
    args : tuple, optional
        Extra arguments passed to the objective function and its
        derivatives (`fun`, `jac` and `hess` functions).
    jac : {callable,  '2-point', '3-point', 'cs', bool}, optional
        Method for computing the gradient vector.
        If it is a callable, it should be a function that returns the gradient
        vector:

            ``jac(x, *args) -> array_like, shape (n,)``

        where ``x`` is an array with shape (n,) and ``args`` is a tuple with
        the fixed parameters. If `jac` is a Boolean and is True, `fun` is
        assumed to return a tuple ``(f, g)`` containing the objective
        function and the gradient.
        If None or False, the gradient will be estimated using 2-point finite
        difference estimation with an absolute step size.
        Alternatively, the keywords  {'2-point', '3-point', 'cs'} can be used
        to select a finite difference scheme for numerical estimation of the
        gradient with a relative step size. These finite difference schemes
        obey any specified `bounds`.
    update_fun_def: Optional[Callable]
        Function to update the gradient sequence. This is an experimental feature to
        allow changing the objective function definition on the fly. In the first place
        this functionality is dedicated to regularized problems for which the
        regularization weight is computed while optimizing the cost function. In order
        to get a hessian matching the new definition of `fun`, the gradient sequence
        must be updated.

            ``update_fun_def(x, f0, grad, x_deque, grad_deque)
            -> f0, grad, updated grad_deque``

    bounds : sequence or `Bounds`, optional
        Bounds on variables for Nelder-Mead, L-BFGS-B, TNC, SLSQP, Powell, and
        trust-constr methods. There are two ways to specify the bounds:

            1. Instance of `Bounds` class.
            2. Sequence of ``(min, max)`` pairs for each element in `x`. None
               is used to specify no bound.
    maxcor : int
        The maximum number of variable metric corrections used to
        define the limited memory matrix. (The limited memory BFGS
        method does not store the full hessian but uses this many terms
        in an approximation to it.)
    ftol : float
        The iteration stops when ``(f^k -
        f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= ftol``.
    gtol : float
        The iteration will stop when ``max{|proj g_i | i = 1, ..., n}
        <= gtol`` where ``pg_i`` is the i-th component of the
        projected gradient.
    eps : float or ndarray
        If `jac is None` the absolute step size used for numerical
        approximation of the jacobian via forward differences.
    maxfun : int
        Maximum number of function evaluations. Note that this function
        may violate the limit because of evaluating gradients by numerical
        differentiation.
        Note that interruptions due to maxfun are postponed
        until the completion of a minimization iteration, consequently it might
        stop after maxfun has been reached.
    maxiter : int
        Maximum number of iterations.
    iprint : int, optional
        Controls the frequency of output. ``iprint < 0`` means no output;
        ``iprint = 0``    print only one line at the last iteration;
        ``0 < iprint < 99`` print also f and ``|proj g|`` every iprint iterations;
        ``iprint >= 99``   print details of every iteration except n-vectors;
    callback : callable, optional
        Called after each iteration. It is a callable with
        the signature:

            ``callback(xk, OptimizeResult state) -> bool``

        where ``xk`` is the current parameter vector. and ``state``
        is an `OptimizeResult` object, with the same fields
        as the ones from the return. If callback returns True
        the algorithm execution is terminated.
    maxls : int, optional
        Maximum number of line search steps (per iteration). Default is 20.
    finite_diff_rel_step : None or array_like, optional
        If `jac in ['2-point', '3-point', 'cs']` the relative step size to
        use for numerical approximation of the jacobian. The absolute step
        size is computed as ``h = rel_step * sign(x) * max(1, abs(x))``,
        possibly adjusted to fit into the bounds. For ``method='3-point'``
        the sign of `h` is ignored. If None (default) then step is selected
        automatically.
    max_steplength: float
        Maximum steplength allowed. The default is 1e8.
    ftol_linesearch: float, optional
        Specify a nonnegative tolerance for the sufficient decrease condition in
        `minpack2.dcsrch <https://ftp.mcs.anl.gov/pub/MINPACK-2/csrch/dcsrch.f>`_
        (used for the line search). This is :math:`c_1` in
        the Armijo condition (or Goldstein, Goldstein-Armijo condition) where
        :math:`\alpha_{k}` is the estimated step.

        .. math::

            f(\mathbf{x}_{k}+\alpha_{k}\mathbf{p}_{k})\leq
            f(\mathbf{x}_{k})+c_{1}\alpha_{k}\mathbf{p}_{k}^{\mathrm{T}}
            \nabla f(\mathbf{x}_{k})

        Note that :math:`0 < c_1 < 1`. Usually :math:`c_1` is small, see the Wolfe
        conditions in :cite:t:`nocedalNumericalOptimization1999`.
        In the fortran implementation
        algo 778, it is hardcoded to 1e-3. The default is 1e-4.
    gtol_linesearch: float, optional
        Specify a nonnegative tolerance for the curvature condition in
        `minpack2.dcsrch <https://ftp.mcs.anl.gov/pub/MINPACK-2/csrch/dcsrch.f>`_
        (used for the line search). This is :math:`c_2` in
        the Armijo condition (or Goldstein, Goldstein-Armijo condition) where
        :math:`\alpha_{k}` is the estimated step.

        .. math::

            \left|\mathbf{p}_{k}^{\mathrm {T}}\nabla f(\mathbf{x}_{k}+\alpha_{k}
            \mathbf{p}_{k})\right|\leq c_{2}\left|\mathbf {p}_{k}^{\mathrm{T}}\nabla
            f(\mathbf{x}_{k})\right|

        Note that :math:`0 < c_1 < c_2 < 1`. Usually, :math:`c_2` is
        much larger than :math:`c_2`.
        see :cite:t:`nocedalNumericalOptimization1999`. In the fortran implementation
        algo 778, it is hardcoded to 0.9. The default is 0.9.
    xtol_linesearch: float, optional
        Specify a nonnegative relative tolerance for an acceptable step in the line
        search procedure (see
        `minpack2.dcsrch <https://ftp.mcs.anl.gov/pub/MINPACK-2/csrch/dcsrch.f>`_).
        In the fortran implementation algo 778, it is hardcoded to 0.1.
        The default is 1e-5.
        See :func:`line_search` parameters.
    eps_SY: float
        Parameter used for updating the L-BFGS matrices. The default is 2.2e-16.

    Returns
    -------
    OptimizeResult
        Wrapper for optimization results (from scipy).

    References
    ----------
    * R. H. Byrd, P. Lu and J. Nocedal. A Limited Memory Algorithm for Bound
      Constrained Optimization, (1995), SIAM Journal on Scientific and
      Statistical Computing, 16, 5, pp. 1190-1208.
    * C. Zhu, R. H. Byrd and J. Nocedal. L-BFGS-B: Algorithm 778: L-BFGS-B,
      FORTRAN routines for large scale bound constrained optimization (1997),
      ACM Transactions on Mathematical Software, 23, 4, pp. 550 - 560.
    * J.L. Morales and J. Nocedal. L-BFGS-B: Remark on Algorithm 778: L-BFGS-B,
      FORTRAN routines for large scale bound constrained optimization (2011),
      ACM Transactions on Mathematical Software, 38, 1.
    """
    lb, ub = get_bounds(x0, bounds)
    max_steplength_user: float = copy.copy(max_steplength)

    # applying the bounds to the initial guess x0
    n = x0.size
    x = clip2bounds(x0, lb, ub)

    Status.START
    # Some display about the problem at hand. The display depends on the value of iprint
    display_start(
        np.finfo(float).eps, n, maxcor, count_var_at_bounds(x, lb, ub), iprint
    )

    # Deque = similar to list but with faster operations to remove and add
    # values to extremities
    X: Deque[NDArrayFloat] = deque()
    G: Deque[NDArrayFloat] = deque()

    # Initialize lbfgsb matrices ("Wa", "Ws", "Wy", "Sy", "Ss", "Wn", "Wt", "Snd")
    mats = LBFGSmat(x.size, maxcor)

    # search direction for the minimization problem
    W: NDArrayFloat = np.zeros([n, 1])
    M: NDArrayFloat = np.zeros([1, 1])
    invMlt: NDArrayFloat = np.zeros([1, 1])
    theta = 1

    # wrapper storing the calls to f and g and handling finite difference approximation
    sf = _prepare_scalar_function(
        fun,
        x0,
        jac=jac,
        args=args,
        epsilon=eps,
        bounds=bounds,
        finite_diff_rel_step=finite_diff_rel_step,
    )

    f0, grad = sf.fun_and_grad(x)

    # Store first res to X and G
    X.append(x0)
    G.append(grad)
    n_iterations = 0

    task_str = "START"
    is_sucess = False
    warnflag = 2

    # For now the free variables at the cauchy points is an empty set
    free_vars = np.array([], dtype=np.int_)

    # Check the infinity norm of the projected gradient
    sbgnrm = projgr(x, grad, lb, ub)
    display_iter(n_iterations, sbgnrm, f0, iprint)

    # Note that interruptions due to maxfun are postponed
    # until the completion of the current minimization iteration.
    while (
        projgr(x, grad, lb, ub) > gtol and n_iterations < max_iter and sf.nfev < maxfun
    ):
        if iprint > 99:
            print(f"\nITERATION {n_iterations + 1}\n")

        oljac0 = f0
        x.copy()
        grad.copy()

        # find cauchy point
        # TODO: replace dictCP by a class
        dictCP = get_cauchy_point(
            x,
            grad,
            lb,
            ub,
            W,
            M,
            invMlt,
            mats,
            theta,
            len(X),
            maxcor,
            iprint,
            n_iterations,
        )

        # Get the free variables for the GCP

        free_vars, Z, A = freev(dictCP["xc"], lb, ub, free_vars, iprint, n_iterations)

        # if n_iterations != 0 and dictCP["free_vars"] != 0:
        # Factorization of the matrix K used in the subspace minimization
        # TODO: there is something I don't get here...
        # K: NDArrayFloat = formk(X, G, Z, A, theta)
        K = None

        # subspace minimization: find the search direction for the minimization problem
        xbar: NDArrayFloat = direct_primal_subspace_minimization(
            x,
            dictCP["xc"],
            free_vars,
            Z,
            dictCP["c"],
            grad,
            lb,
            ub,
            W,
            M,
            theta,
            K,
        )
        d = xbar - x

        # TODO: implement
        # - Primal Conjugate Gradient Method (section 5.2 in Byrd et al. (1995))
        # - Dual Method for Subspace Minimization (section 5.3 in Byrd et al. (1995)

        # max_stpl = computer defined
        # max_steplength = user defined
        max_stpl: float = max_allowed_steplength(x, d, lb, ub, max_steplength_user)
        steplength = line_search(
            x,
            f0,
            grad,
            d,
            n_iterations,
            max_stpl,
            sf.fun_and_grad,
            ftol_linesearch,
            gtol_linesearch,
            xtol_linesearch,
            maxls,
            iprint,
        )

        if steplength is None:
            if len(X) == 0:
                # Hessian already rebooted: abort.
                task_str = "Error: cannot compute new steplength : abort"
                f, grad = sf.fun_and_grad(x)
                warnflag = 2
                is_sucess = False
                break
            else:
                # Reboot BFGS-Hessian:
                X.clear()
                G.clear()
                W = np.zeros([n, 1])
                M = np.zeros([1, 1])
                theta = 1
        else:
            # x update
            x += steplength * d

            f0, grad = sf.fun_and_grad(x)

            # perform a potential update of the objective function definition and
            # upgrade the gradient and the past sequence of gradients accordingly
            if update_fun_def is not None:
                f0, grad, G = update_fun_def(x, f0, grad, X, G)

            W, M, invMlt, theta = get_lbfgs_matrices(
                x.copy(),  # copy otherwise x might be changed in X when updated
                grad,
                X,
                G,
                maxcor,
                mats,
                W.copy(),
                M.copy(),
                invMlt,
                copy.copy(theta),
                False,
                eps_SY,
            )

            if (oljac0 - f0) / max(abs(oljac0), abs(f0), 1) < ftol:
                task_str = "CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH"
                is_sucess = True
                warnflag = 0
                break

            # callback is a user defined mechanism to stop optimization
            # if callback returns True, then it stops.
            if callback is not None:
                if callback(
                    np.copy(x),
                    OptimizeResult(
                        fun=f0,
                        jac=grad,
                        nfev=sf.nfev,
                        njev=sf.ngev,
                        nit=n_iterations,
                        status=warnflag,
                        message=task_str,
                        x=x,
                        success=is_sucess,
                        hess_inv=LbfgsInvHessProduct(
                            np.diff(np.array(X), axis=0), np.diff(np.array(G), axis=0)
                        ),
                    ),
                ):
                    task_str = "STOP: USER CALLBACK"
                    is_sucess = True
                    break

            # Result display
            display_results(
                iprint, n_iterations, max_iter, x, grad, lb, ub, f0, gtol, False
            )

            display_iter(n_iterations + 1, sbgnrm, f0, iprint)

            n_iterations += 1

    # Final display
    display_results(iprint, n_iterations, max_iter, x, grad, lb, ub, f0, gtol, True)

    if projgr(x, grad, lb, ub) <= gtol:
        task_str = "CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL"
        is_sucess = True
        warnflag = 1
    if n_iterations == max_iter:
        task_str = "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT"
        is_sucess = True
        warnflag = 1
    elif sf.nfev >= maxfun:
        task_str = "STOP: TOTAL NO. of f AND g EVALUATIONS EXCEEDS LIMIT"
        is_sucess = True
        warnflag = 1

    # error: b'ERROR: STPMAX .LT. STPMIN'

    return OptimizeResult(
        fun=f0,
        jac=grad,
        nfev=sf.nfev,
        njev=sf.ngev,
        nit=n_iterations,
        status=warnflag,
        message=task_str,
        x=x,
        success=is_sucess,
        hess_inv=LbfgsInvHessProduct(
            np.diff(np.array(X), axis=0), np.diff(np.array(G), axis=0)
        ),
    )
