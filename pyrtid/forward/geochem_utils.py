# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 Antoine COLLET

"""
Numerical helpers for the local (cell-wise) geochemical solvers.

The module gathers the generic building blocks used to solve the small non-linear
chemical system of each grid cell with a Newton-Raphson algorithm:

- :func:`newton`: a damped Newton-Raphson loop with a bounded number of iterations.
- :func:`backtracking_linesearch`: a dependency-free Armijo backtracking line search
  (section 9.7.1 of *Numerical Recipes*).
- :func:`standalone_linesearch`: an optional Wolfe line search relying on the
  ``lbfgsb`` package (imported lazily, so that ``lbfgsb`` is not a hard dependency).
- :func:`get_polish`: a per-component step damping ("polishing") that prevents
  concentrations from changing sign during an iteration.
- :func:`solve_with_svd`: a truncated-SVD (minimum norm) linear solver, robust to
  (nearly) singular Jacobians.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
from scipy import linalg
from scipy.optimize import OptimizeResult

from pyrtid.utils import NDArrayFloat

__all__ = [
    "SMALL_VALUE",
    "backtracking_linesearch",
    "standalone_linesearch",
    "get_polish",
    "solve_with_svd",
    "newton",
]

#: Values below this threshold are considered null when computing relative changes.
SMALL_VALUE = 1e-25


def backtracking_linesearch(
    fun: Callable[[NDArrayFloat], float],
    x: NDArrayFloat,
    d: NDArrayFloat,
    f0: float,
    slope: float,
    alpha_max: float = 1.0,
    c1: float = 1e-4,
    max_iter: int = 30,
    min_shrink: float = 0.1,
    max_shrink: float = 0.5,
) -> float | None:
    r"""
    Find a step length satisfying the Armijo (sufficient decrease) condition.

    The step is found by backtracking from ``alpha_max`` using a safeguarded
    quadratic interpolation, see section 9.7.1 *Line Searches and Backtracking* of
    W. H. Press and S. A. Teukolsky, *Numerical Recipes 3rd Edition*, Cambridge
    University Press, 2007. The returned ``alpha`` satisfies

    .. math::
        f(x + \alpha d) \leq f(x) + c_1 \alpha \langle \nabla f(x), d \rangle.

    Parameters
    ----------
    fun : Callable[[NDArrayFloat], float]
        Objective function.
    x : NDArrayFloat
        Current point.
    d : NDArrayFloat
        Search direction. It must be a descent direction.
    f0 : float
        Objective function value at ``x``.
    slope : float
        Directional derivative :math:`\langle \nabla f(x), d \rangle` at ``x``.
        It must be strictly negative.
    alpha_max : float, optional
        Largest step length tried first, by default 1.0 (full Newton step).
    c1 : float, optional
        Sufficient decrease parameter, with :math:`0 < c_1 < 1`. By default 1e-4.
    max_iter : int, optional
        Maximum number of objective evaluations, by default 30.
    min_shrink : float, optional
        A new trial step is never smaller than ``min_shrink`` times the previous
        one, by default 0.1.
    max_shrink : float, optional
        A new trial step is never larger than ``max_shrink`` times the previous
        one, by default 0.5.

    Returns
    -------
    Optional[float]
        The step length, or None if ``d`` is not a descent direction or if no
        acceptable step has been found within ``max_iter`` evaluations.
    """
    if not slope < 0.0:
        return None

    alpha = alpha_max
    for _ in range(max_iter):
        f_new = fun(x + alpha * d)
        if np.isfinite(f_new) and f_new <= f0 + c1 * alpha * slope:
            return alpha

        if np.isfinite(f_new):
            # minimizer of the quadratic interpolation of f along d
            denom = 2.0 * (f_new - f0 - slope * alpha)
            alpha_new = -slope * alpha**2 / denom if denom > 0.0 else 0.0
        else:
            alpha_new = 0.0
        alpha = min(max(alpha_new, min_shrink * alpha), max_shrink * alpha)
    return None


def standalone_linesearch(
    x0: NDArrayFloat,
    fun: Callable,
    grad: Callable,
    d: NDArrayFloat,
    bounds: NDArrayFloat | None = None,
    max_steplength_user: float = 1e8,
    ftol: float = 1e-3,
    gtol: float = 0.9,
    xtol: float = 1e-1,
    max_iter: int = 30,
    opt_iter: int = 0,
    iprint: int = 10,
    logger: logging.Logger | None = None,
) -> tuple[float | None, int, int, float, float, NDArrayFloat]:
    r"""
    Find a step satisfying the strong Wolfe conditions (``lbfgsb`` line search).

    The step satisfies both a sufficient decrease condition

    .. math::
        f(x_0 + \alpha d) \leq f(x_0) + c_1 \alpha \langle \nabla f(x_0), d \rangle,

    and a curvature condition

    .. math::
        |\langle \nabla f(x_0 + \alpha d), d \rangle| \leq c_2
        |\langle \nabla f(x_0), d \rangle|.

    If :math:`c_1 < c_2` and the function is bounded below, there is always a step
    satisfying both conditions.

    Note
    ----
    This function relies on the (internal) line search of the ``lbfgsb`` package,
    which is imported lazily. For a dependency-free alternative, use
    :func:`backtracking_linesearch`.

    Parameters
    ----------
    x0 : NDArrayFloat
        Starting point.
    fun : Callable
        Objective function.
    grad : Callable
        Gradient of the objective function.
    d : NDArrayFloat
        Search direction.
    bounds : sequence or `Bounds`, optional
        Bounds on the variables, as a sequence of ``(min, max)`` pairs for each
        element of ``x0`` (None is used to specify no bound) or as a ``Bounds``
        instance. By default None (unbounded).
    max_steplength_user : float, optional
        Maximum step length allowed. The default is 1e8 (no practical limit).
    ftol : float, optional
        Tolerance for the sufficient decrease condition (:math:`c_1`, with
        :math:`0 < c_1 < c_2 < 1`). By default 1e-3, as hardcoded in algorithm 778.
    gtol : float, optional
        Tolerance for the curvature condition (:math:`c_2`). By default 0.9, as
        hardcoded in algorithm 778.
    xtol : float, optional
        Relative tolerance for an acceptable step. By default 1e-1, as hardcoded in
        algorithm 778.
    max_iter : int, optional
        Maximum number of line search iterations, by default 30.
    opt_iter : int, optional
        Number of iterations of the outer optimization algorithm, by default 0.
    iprint : int, optional
        Controls the frequency of output. ``iprint < 0`` means no output;
        ``iprint = 0`` prints only one line at the last iteration;
        ``0 < iprint < 99`` prints also f and ``|proj g|`` every iprint iterations;
        ``iprint >= 99`` prints details of every iteration except n-vectors.
    logger : Optional[logging.Logger], optional
        Logger instance. If None, nothing is displayed, no matter the value of
        `iprint`.

    Returns
    -------
    alpha : float or None
        Step length such that ``x_new = x0 + alpha * d``, or None if the line search
        did not converge.
    nfev : int
        Number of function evaluations made.
    ngev : int
        Number of gradient evaluations made.
    new_fval : float
        New function value ``f(x0 + alpha * d)``. Equal to ``f0`` on failure.
    old_fval : float
        Old function value ``f(x0)``.
    new_grad : NDArrayFloat
        Gradient at ``x0 + alpha * d`` (or the search direction ``d`` on failure).
    """
    # Lazy imports: the lbfgsb internals are only needed by this function and they
    # may change from one version to another.
    from lbfgsb.base import get_bounds, is_any_inf
    from lbfgsb.linesearch import line_search as ls2
    from lbfgsb.scalar_function import ScalarFunction

    # The internals are not typed consistently from one version to another
    _get_bounds: Any = get_bounds
    _line_search: Any = ls2
    _scalar_function: Any = ScalarFunction

    lb, ub = _get_bounds(x0, bounds)

    sf_kwargs = dict(
        fun=fun,
        x0=x0,
        grad=grad,
        finite_diff_bounds=(lb, ub),
        finite_diff_rel_step=None,
    )
    try:
        sf = _scalar_function(args=(), **sf_kwargs)
    except TypeError:  # newer versions of lbfgsb dropped the `args` argument
        sf = _scalar_function(**sf_kwargs)
    f0 = sf.fun(x0)

    alpha = _line_search(
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
    Get a (per-component) polishing factor for a Newton increment.

    The factor damps the components of the increment which are large compared to the
    current values, so that concentrations do not change sign. See section 10.3.2 of
    Yann's report *Improvement of the Newton-Raphson method*.

    Parameters
    ----------
    dC : NDArrayFloat
        Newton increment, to be *subtracted* from ``C``.
    C : NDArrayFloat
        Current (non-preconditioned) unknowns.

    Returns
    -------
    NDArrayFloat
        Polishing factors, with the same shape as ``C``. They are all equal to one
        if at least one component of ``C`` is null (relative changes are then
        undefined).
    """
    pf: NDArrayFloat = np.ones_like(C, dtype=float)  # polishing factor (default: 1)
    if not np.all(np.abs(C) > SMALL_VALUE):
        return pf

    a = 0.5
    b = 3.0
    c = 0.9

    ratio: NDArrayFloat = dC / C
    abs_ratio: NDArrayFloat = np.abs(ratio)
    mask = abs_ratio > a
    if not np.any(mask):
        return pf

    # Only evaluate the formulas where they are used (ratio > a > 0 => ratio != 0)
    r = ratio[mask]
    ar = abs_ratio[mask]
    pf[mask] = np.where(
        r > 0.0,
        (b * ar - a * a) / ((b + ar - 2.0 * a) * r),
        -c * (ar - a * a) / ((1.0 + ar - 2.0 * a) * r),
    )
    return pf


def solve_with_svd(
    A: NDArrayFloat,
    b: NDArrayFloat,
    atol: float | None = None,
    rtol: float | None = None,
    check_finite: bool = True,
) -> NDArrayFloat:
    """
    Solve the system ``A x = b`` with a truncated SVD (minimum-norm solution).

    Singular values below ``atol + rtol * max(s)`` are discarded. For a full-rank
    square matrix, the result is the usual solution of the linear system. For a
    singular or rectangular matrix, it is the minimum-norm least-squares solution
    (:math:`x = A^{+} b`).

    Parameters
    ----------
    A : NDArrayFloat
        Matrix, not necessarily square, with shape (M, N).
    b : NDArrayFloat
        Right-hand side, with shape (M,) or (M, K).
    atol : Optional[float], optional
        Absolute threshold below which singular values are discarded.
        The default is 0.
    rtol : Optional[float], optional
        Relative threshold (to the largest singular value) below which singular
        values are discarded. The default is ``max(M, N) * eps``.
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        By default True.

    Returns
    -------
    NDArrayFloat
        The solution ``x`` with shape (N,) or (N, K).

    Raises
    ------
    ValueError
        If ``atol`` or ``rtol`` is negative.
    """
    atol = 0.0 if atol is None else atol
    u, s, vh = linalg.svd(A, full_matrices=False, check_finite=check_finite)
    rtol = max(A.shape) * np.finfo(u.dtype).eps if rtol is None else rtol

    if (atol < 0.0) or (rtol < 0.0):
        raise ValueError("atol and rtol values must be positive.")

    rank = int(np.count_nonzero(s > atol + np.max(s, initial=0.0) * rtol))
    if rank == 0:
        return np.zeros((A.shape[1],) + np.shape(b)[1:], dtype=u.dtype)

    coeffs = u[:, :rank].T @ b
    coeffs = coeffs / (s[:rank] if coeffs.ndim == 1 else s[:rank, None])
    return vh[:rank].T @ coeffs


def newton(
    x0: NDArrayFloat,
    get_res: Callable[[NDArrayFloat], NDArrayFloat],
    get_invjacres: Callable[[NDArrayFloat], NDArrayFloat],
    atol: float,
    linesearch: Callable[[NDArrayFloat, NDArrayFloat, int], object] | None = None,
    rtol: float = 1e-12,
    max_iter: int = 100,
) -> OptimizeResult:
    r"""
    Find a root of :math:`F(x) = 0` with a (damped) Newton-Raphson algorithm.

    The iteration reads :math:`x^{k+1} = x^k - \alpha^k J^{-1}(x^k) F(x^k)` where
    :math:`\alpha^k` is a scalar or a per-component damping factor given by the
    optional line search. The algorithm stops when

    - the norm of the residuals is below ``atol``, or
    - the (undamped) Newton increment is below ``atol + rtol * ||x||``, or
    - ``max_iter`` iterations have been performed (``success`` is then False).

    Parameters
    ----------
    x0 : NDArrayFloat
        Initial guess of the unknowns.
    get_res : Callable[[NDArrayFloat], NDArrayFloat]
        Function returning the residuals :math:`F(x)`.
    get_invjacres : Callable[[NDArrayFloat], NDArrayFloat]
        Function returning the Newton increment :math:`J^{-1}(x) F(x)`, which is
        subtracted from ``x``.
    atol : float
        Absolute tolerance on the residuals and on the Newton increment.
    linesearch : Optional[Callable[[NDArrayFloat, NDArrayFloat, int], object]]
        Function ``linesearch(x, dx, n_iterations)`` returning the damping factor
        (a float or an array broadcastable to ``x``) to apply to the increment
        ``dx``. If None, the full Newton step is taken. By default None.
    rtol : float, optional
        Relative tolerance on the Newton increment, by default 1e-12.
    max_iter : int, optional
        Maximum number of iterations, by default 100.

    Returns
    -------
    OptimizeResult
        The result with attributes ``x`` (solution), ``success`` (bool),
        ``status`` (``"convergence"`` or ``"max_iter"``), ``message``, ``nit``
        (number of iterations) and ``fun`` (norm of the final residuals).
    """
    x = np.array(x0, dtype=float)  # copy
    residuals = get_res(x)
    res_norm = float(np.linalg.norm(residuals))
    n_iterations = 0
    is_converged = res_norm <= atol

    while not is_converged and n_iterations < max_iter:
        n_iterations += 1

        # Newton increment (to subtract) and optional damping
        dx = get_invjacres(x)
        alpha = np.asarray(1.0)
        if linesearch is not None:
            # scalar or per-component damping factor
            alpha = np.asarray(linesearch(x, dx, n_iterations), dtype=float)

        # update x and the residuals
        x = x - alpha * dx
        residuals = get_res(x)
        res_norm = float(np.linalg.norm(residuals))

        is_converged = (res_norm <= atol) or (
            float(np.linalg.norm(dx)) <= atol + rtol * float(np.linalg.norm(x))
        )

    return OptimizeResult(
        x=x,
        success=is_converged,
        status="convergence" if is_converged else "max_iter",
        message=(
            "Newton converged."
            if is_converged
            else f"Newton did not converge within {max_iter} iterations."
        ),
        nit=n_iterations,
        fun=res_norm,
    )
