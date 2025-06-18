import logging
from typing import Callable, Optional, Tuple

import numpy as np  # NumPy for numerical operations
import scipy as sp
from lbfgsb.base import get_bounds, is_any_inf
from lbfgsb.linesearch import line_search as ls2
from lbfgsb.scalar_function import ScalarFunction
from lbfgsb.types import NDArrayFloat
from scipy.optimize import OptimizeResult

SMALL_VALUE = 1e-25


def standalone_linesearch(
    x0: NDArrayFloat,
    fun: Callable,
    grad: Callable,
    d: NDArrayFloat,
    bounds: Optional[NDArrayFloat] = None,
    max_steplength_user: float = 1e-8,
    ftol: float = 1e-3,
    gtol: float = 0.9,
    xtol: float = 1e-1,
    max_iter: int = 30,
    opt_iter: int = 0,
    iprint: int = 10,
    logger: Optional[logging.Logger] = None,
) -> Tuple[Optional[float], int, int, float, float, NDArrayFloat]:
    r"""
    Find a step that satisfies both decrease condition and a curvature condition.

        f(x0+stp*d) <= f(x0) + alpha*stp*\langle f'(x0),d\rangle,

    and the curvature condition

        abs(f'(x0+stp*d)) <= beta*abs(\langle f'(x0),d\rangle).

    If alpha is less than beta and if, for example, the functionis bounded below, then
    there is always a step which satisfies both conditions.

    Note
    ----
    When using scipy-1.11 and below, this subroutine calls subroutine dcsrch from the
    Minpack2 library to perform the line search.  Subroutine dscrch is safeguarded so
    that all trial points lie within the feasible region. Otherwise, it uses the
    python reimplementation introduced in scipy-1.12.

    Parameters
    ----------
    x0 : NDArrayFloat
        Starting point.
    fun : Callable
        Objective function.
    grad : Callable
        Gradient of the objective function.
    bounds : sequence or `Bounds`, optional
        Bounds on variables for Nelder-Mead, L-BFGS-B, TNC, SLSQP, Powell, and
        trust-constr methods. There are two ways to specify the bounds:

            1. Instance of `Bounds` class.
            2. Sequence of ``(min, max)`` pairs for each element in `x`. None
               is used to specify no bound.
    d : NDArrayFloat
        Search direction.
    max_steplength : float
        Maximum steplength allowed.
    ftol: float, optional
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
    gtol: float, optional
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
    xtol: float, optional
        Specify a nonnegative relative tolerance for an acceptable step in the line
        search procedure (see
        `minpack2.dcsrch <https://ftp.mcs.anl.gov/pub/MINPACK-2/csrch/dcsrch.f>`_).
        In the fortran implementation algo 778, it is hardcoded to 0.1.
        The default is 1e-5.
    max_iter : int, optional
            Maximum number of linesearch iterations, by default 30.
    iprint : int, optional
        Controls the frequency of output. ``iprint < 0`` means no output;
        ``iprint = 0``    print only one line at the last iteration;
        ``0 < iprint < 99`` print also f and ``|proj g|`` every iprint iterations;
        ``iprint >= 99``   print details of every iteration except n-vectors;
    logger: Optional[Logger], optional
        :class:`logging.Logger` instance. If None, nothing is displayed, no matter the
        value of `iprint`, by default None.

    Returns
    -------
    alpha : float or None
        Alpha for which ``x_new = x0 + alpha * pk``,
        or None if the line search algorithm did not converge.
    fc : int
        Number of function evaluations made.
    gc : int
        Number of gradient evaluations made.
    new_fval : float or None
        New function value ``f(x_new)=f(x0+alpha*pk)``,
        or None if the line search algorithm did not converge.
    old_fval : float
        Old function value ``f(x0)``.
    new_slope : float or None
        The local slope along the search direction at the
        new value ``<myfprime(x_new), pk>``,
        or None if the line search algorithm did not converge.
    """
    lb, ub = get_bounds(x0, bounds)

    sf = ScalarFunction(
        fun=fun,
        x0=x0,
        args=(),
        grad=grad,
        finite_diff_bounds=(lb, ub),
        finite_diff_rel_step=None,
    )
    f0 = sf.fun(x0)

    alpha = ls2(
        x0=x0,
        f0=f0,
        g0=grad(x0),
        d=d,
        lb=lb,
        ub=ub,
        is_boxed=not is_any_inf([lb, ub]),
        sf=sf,
        above_iter=opt_iter,
        max_steplength_user=max_steplength_user,
        ftol=ftol,
        gtol=gtol,
        xtol=xtol,
        max_iter=max_iter,
        iprint=iprint,
        logger=logger,
    )
    if alpha is None:
        return (None, sf.nfev, sf.ngev, f0, f0, d)
    x_new = x0 + alpha * d
    return (alpha, sf.nfev, sf.ngev, sf.fun(x_new), f0, grad(x_new))


def get_polish(dC: NDArrayFloat, C: NDArrayFloat) -> NDArrayFloat:
    """
    Get a polishing factor.

    See section 10.3.2 of Yann's report Improvement of the Newton-Raphson method.
    """
    ratio: NDArrayFloat = dC / C
    pf: NDArrayFloat = np.ones_like(ratio)  # polishing factor initialized to one
    abs_ratio: NDArrayFloat = np.abs(ratio)
    a = 0.5
    b = 3.0
    c = 0.9

    if all(x > SMALL_VALUE for x in abs(C)):
        tmp = np.where(
            ratio > 0.0,
            (b * abs_ratio - a * a) / ((b + abs_ratio - 2.0 * a) * ratio),
            -c * (abs_ratio - a * a) / ((1.0 + abs_ratio - 2.0 * a) * ratio),
        )
        mask = abs_ratio > a
        pf[mask] = tmp[mask]
    else:
        pf = np.ones_like(ratio)

    return pf


def solve_with_svd(
    A: NDArrayFloat,
    b: NDArrayFloat,
    atol: Optional[float] = None,
    rtol: Optional[float] = None,
    check_finite: bool = True,
) -> NDArrayFloat:
    """
    Solve the system Ax = b using the SVD as a preconditioner.

    Parameters
    ----------
    A : NDArrayFloat
        Matrix, not necessarily square.
    b : NDArrayFloat
        RHS vector.
    rcond : _type_, optional
        Threshold to discard small singular values, by default 1e-15.

    Returns
    -------
    NDArrayFloat
        x = A^{1} @ b
    """
    u, s, vh = sp.linalg.svd(A, full_matrices=False, check_finite=check_finite)
    t = u.dtype.char.lower()
    maxS = np.max(s, initial=0.0)

    atol = 0.0 if atol is None else atol
    rtol = max(A.shape) * np.finfo(t).eps if (rtol is None) else rtol

    if (atol < 0.0) or (rtol < 0.0):
        raise ValueError("atol and rtol values must be positive.")

    val = atol + maxS * rtol
    rank = np.sum(s > val)

    u = u[:, :rank]
    u /= s[:rank]
    vh = vh[:rank]

    def invA_matvec(z: NDArrayFloat) -> NDArrayFloat:
        return vh.T @ u.T @ z

    # solve P^{-1} A x = P^{-1} b
    # instead of A x = b
    # The advantage is that P^{-1} A is square even if A is not square
    # print(invA_matvec(A))  # this should be close to the identity matrix
    return sp.linalg.solve(invA_matvec(A), invA_matvec(b))


def newton(
    x0: NDArrayFloat,
    get_res: Callable,
    get_invjacres: Callable,
    atol: float,
    linesearch: Optional[Callable] = None,
) -> OptimizeResult:
    """
    Solve Ax = b with Newton.

    Parameters
    ----------
    x0 : NDArrayFloat
        Unknowns to be found. With shape (Ns, Ne).
    get_res : Callable
        _description_
    get_invjacres : Callable
        _description_
    atol : float
        _description_
    linesearch : Optional[Callable], optional
        _description_, by default None

    Returns
    -------
    OptimizeResult
        _description_
    """

    # number of iterations
    n_iterations = 0
    # make sure that the residuals norm and update_steps norm are above atol
    residuals_norm = atol + 10.0
    dt_norm = atol + 10.0
    # initiate x as x0
    x = x0.copy()

    while (residuals_norm > atol) and (dt_norm > atol):
        # update the number of iterations
        n_iterations += 1

        # print(f"iteration #{n_iteration}")

        # Calculate the residual error
        residuals = get_res(x)

        # compute the update
        dx = get_invjacres(x)

        # eventually use a linesearch step
        if linesearch is not None:
            alpha = linesearch(x, dx, n_iterations)
        else:
            alpha = 1.0

        # update x
        x = x - alpha * dx

        # update the norms
        residuals_norm = np.linalg.norm(residuals).item()
        residuals_norm = np.linalg.norm(dx).item()

    return OptimizeResult(
        x=x,
        success=True,
        status="convergence",
        nit=n_iterations,
    )
