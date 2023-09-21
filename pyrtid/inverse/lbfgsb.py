import copy
from collections import deque
from typing import Callable, Deque, Dict, Optional, Tuple

import numpy as np
from scipy.optimize import minpack2
from scipy.optimize._constraints import old_bound_to_new
from scipy.optimize._lbfgsb_py import LbfgsInvHessProduct  # noqa : F401
from scipy.optimize._optimize import _check_unknown_options  # noqa : F401
from scipy.optimize._optimize import OptimizeResult, _prepare_scalar_function

from pyrtid.utils import NDArrayFloat


def compute_Cauchy_point(
    x: NDArrayFloat,
    grad: NDArrayFloat,
    lb: NDArrayFloat,
    ub: NDArrayFloat,
    W: NDArrayFloat,
    M: NDArrayFloat,
    theta: float,
):
    """
    Computes the generalized Cauchy point (GCP).

    It is defined as the first local minimizer of the quadratic

    .. math::
    :nowrap:
        \[\langle grad,s\rangle + \frac{1}{2} \langle s,
        (\theta I + WMW^\intercal)s\rangle\]

    along the projected gradient direction
    .. math:: $P_[l,u](x-\theta grad).$

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
    theta : float
        Part of limited memory BFGS Hessian approximation.

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
    eps_f_sec = 1e-30
    t = np.empty(x.size)
    d = np.empty(x.size)
    x_cp = x.copy()

    # TODO: refactor this
    for i in range(x.size):
        if grad[i] < 0:
            t[i] = (x[i] - ub[i]) / grad[i]
        elif grad[i] > 0:
            t[i] = (x[i] - lb[i]) / grad[i]
        else:
            t[i] = np.inf
        if t[i] == 0:
            d[i] = 0
        else:
            d[i] = -grad[i]

    F = np.argsort(t)
    F = [i for i in F if t[i] > 0]
    t_old = 0
    F_i = 0
    b = F[0]
    t_min = t[b]
    Dt = t_min

    p = np.transpose(W).dot(d)
    c = np.zeros(p.size)
    f_prime = -d.dot(d)
    f_second = -theta * f_prime - p.dot(M.dot(p))
    f_sec0 = f_second
    Dt_min = -f_prime / f_second

    while Dt_min >= Dt and F_i < len(F):
        if d[b] > 0:
            x_cp[b] = ub[b]
        elif d[b] < 0:
            x_cp[b] = lb[b]
        x_bcp = x_cp[b]

        zb = x_bcp - x[b]
        c += Dt * p
        W_b = W[b, :]
        g_b = grad[b]

        f_prime += Dt * f_second + g_b * (g_b + theta * zb - W_b.dot(M.dot(c)))
        f_second -= g_b * (g_b * theta + W_b.dot(M.dot(2 * p + g_b * W_b)))
        f_second = min(f_second, eps_f_sec * f_sec0)

        Dt_min = -f_prime / f_second

        p += g_b * W_b
        d[b] = 0
        t_old = t_min
        F_i += 1

        if F_i < len(F):
            b = F[F_i]
            t_min = t[b]
            Dt = t_min - t_old
        else:
            t_min = np.inf

    Dt_min = 0 if Dt_min < 0 else Dt_min
    t_old += Dt_min

    for i in range(x.size):
        if t[i] >= t_min:
            x_cp[i] = x[i] + t_old * d[i]

    F = [i for i in F if t[i] != t_min]

    c += Dt_min * p
    return {"xc": x_cp, "c": c, "F": F}


def minimize_model(
    x: NDArrayFloat,
    xc: NDArrayFloat,
    c: NDArrayFloat,
    grad: NDArrayFloat,
    lb: NDArrayFloat,
    ub: NDArrayFloat,
    W: NDArrayFloat,
    M: NDArrayFloat,
    theta: float,
) -> Dict:
    """
    Computes an approximate solution of the subspace problem.

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

    Returns
    -------
    Dict
        dict containing a computed value of:
        - 'xbar' the minimizer

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

    ### Début de la multiplication avec le Hessien ?

    invThet = 1.0 / theta

    Z = list()
    free_vars = list()
    n = xc.size
    unit = np.zeros(n)
    for i in range(n):
        unit[i] = 1
        if (xc[i] != ub[i]) and (xc[i] != lb[i]):
            free_vars.append(i)
            Z.append(unit.copy())
        unit[i] = 0

    if len(free_vars) == 0:
        return {"xbar": xc}

    Z = np.asarray(Z).T
    WTZ = W.T.dot(Z)

    rHat = [(grad + theta * (xc - x) - W.dot(M.dot(c)))[ind] for ind in free_vars]
    v = WTZ.dot(rHat)
    v = M.dot(v)

    N = invThet * WTZ.dot(np.transpose(WTZ))
    N = np.eye(N.shape[0]) - M.dot(N)
    v = np.linalg.solve(N, v)

    dHat = -invThet * (rHat + invThet * np.transpose(WTZ).dot(v))

    ### Fin de la multiplication avec le Hessien ?

    # Find alpha
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

    return {"xbar": xbar}


def max_allowed_steplength(
    x: NDArrayFloat,
    d: NDArrayFloat,
    lb: NDArrayFloat,
    ub: NDArrayFloat,
    max_steplength: float,
) -> float:
    """
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
    max_stpl = max_steplength
    for i in range(x.size):
        if d[i] > 0:
            max_stpl = min(max_stpl, (ub[i] - x[i]) / d[i])
        elif d[i] < 0:
            max_stpl = min(max_stpl, (lb[i] - x[i]) / d[i])
    return max_stpl


def line_search(
    x0: NDArrayFloat,
    f0: float,
    g0: NDArrayFloat,
    d: NDArrayFloat,
    above_iter: int,
    max_steplength: float,
    fun_and_grad: Callable[[NDArrayFloat], Tuple[float, NDArrayFloat]],
    alpha: float = 1e-4,
    beta: float = 0.9,
    xtol_minpack: float = 1e-5,
    max_iter: int = 30,
) -> float:
    """
    Find a step that satisfies both decrease condition and a curvature condition.

        f(x0+stp*d) <= f(x0) + alpha*stp*\langle f'(x0),d\rangle,

    and the curvature condition

        abs(f'(x0+stp*d)) <= beta*abs(\langle f'(x0),d\rangle).

    If alpha is less than beta and if, for example, the functionis bounded below, then
    there is always a step which satisfies both conditions.

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
    alpha : float, optional
        _description_, by default 1e-4.
    beta : float, optional
        Parameters of the decrease and curvature conditions, by default 0.9.
    xtol_minpack : float, optional
        Tolerance used in minpack2.dcsrch, by default 1e-5.
    max_iter : int, optional
        Maximum number of linesearch iterations, by default 30.

    Returns
    -------
    float
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

    steplength_0 = 1 if max_steplength > 1 else 0.5 * max_steplength
    f_m1 = f0
    dphi = g0.dot(d)
    dphi_m1 = dphi
    i = 0

    if above_iter == 0:
        max_steplength = 1.0
        steplength_0 = min(1.0 / np.sqrt(d.dot(d)), 1.0)

    isave = np.zeros((2,), np.intc)
    dsave = np.zeros((13,), float)
    task = b"START"

    while i < max_iter:
        steplength, f0, dphi, task = minpack2.dcsrch(
            steplength_0,
            f_m1,
            dphi_m1,
            alpha,
            beta,
            xtol_minpack,
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

    return steplength


def get_lbfgs_matrices(
    xk: NDArrayFloat,
    gk: NDArrayFloat,
    X: Deque[NDArrayFloat],
    G: Deque[NDArrayFloat],
    maxcor: int,
    W: NDArrayFloat,
    M: NDArrayFloat,
    thet: float,
    is_force_update: bool,
    eps: float = 2.2e-16,
) -> Tuple[NDArrayFloat, NDArrayFloat, float]:
    """
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

    # TODO: Why do we perform that check ?
    if sTy > eps * yTy:
        is_current_update_accepted = True
        X.append(xk)
        G.append(gk)
        if len(X) > maxcor:
            X.popleft()
            G.popleft()

    # two conditions to update the inverse Hessian approximation
    if is_force_update or is_current_update_accepted:
        # Update the lbfgsb matrices
        Sarray = np.diff(np.array(X), axis=0).T
        Yarray = np.diff(np.array(G), axis=0).T

        STS = np.transpose(Sarray).dot(Sarray)
        L = np.transpose(Sarray).dot(Yarray)
        D = np.diag(-np.diag(L))
        L = np.tril(L, -1)

        thet = yTy / sTy
        W = np.hstack([Yarray, thet * Sarray])
        M = np.linalg.inv(np.hstack([np.vstack([D, L]), np.vstack([L.T, thet * STS])]))

    return W, M, thet


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


def minimize_lbfgsb(
    *,
    x0: NDArrayFloat,
    fun: Callable[[NDArrayFloat, ...], float],
    args: Tuple = (),
    jac=Optional[Callable[[NDArrayFloat, ...], NDArrayFloat]],
    bounds: Optional[NDArrayFloat] = None,
    disp: Optional[int] = None,
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
    alpha_linesearch: float = 1e-4,
    beta_linesearch: float = 0.9,
    max_steplength: float = 1e8,
    xtol_minpack: float = 1e-5,
    eps_SY: float = 2.2e-16,
) -> OptimizeResult:
    """
    Solves bound constrained optimization problems by using the compact formula
    of the limited memory BFGS updates.

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
    bounds : sequence or `Bounds`, optional
        Bounds on variables for Nelder-Mead, L-BFGS-B, TNC, SLSQP, Powell, and
        trust-constr methods. There are two ways to specify the bounds:

            1. Instance of `Bounds` class.
            2. Sequence of ``(min, max)`` pairs for each element in `x`. None
               is used to specify no bound.
    disp : None or int
        If `disp is None` (the default), then the supplied version of `iprint`
        is used. If `disp is not None`, then it overrides the supplied version
        of `iprint` with the behaviour you outlined.
    maxcor : int
        The maximum number of variable metric corrections used to
        define the limited memory matrix. (The limited memory BFGS
        method does not store the full hessian but uses this many terms
        in an approximation to it.)
    ftol : float
        The iteration stops when ``(f^k -
        f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= ftol``.
        # TODO: see what definition is right.
        Tolerance on function change: programs ends when
        (f_k-f_{k+1})/max(|f_k|,|f_{k+1}|,1) < ftol
    gtol : float
        The iteration will stop when ``max{|proj g_i | i = 1, ..., n}
        <= gtol`` where ``pg_i`` is the i-th component of the
        projected gradient.
        # TODO: see what definition is right.
        Tolerance on projected gradient: programs converges when
        P(x-grad, l, u)<gtol.
    eps : float or ndarray
        If `jac is None` the absolute step size used for numerical
        approximation of the jacobian via forward differences.
    maxfun : int
        Maximum number of function evaluations. Note that this function
        may violate the limit because of evaluating gradients by numerical
        differentiation.
    maxiter : int
        Maximum number of iterations.
    iprint : int, optional
        Controls the frequency of output. ``iprint < 0`` means no output;
        ``iprint = 0``    print only one line at the last iteration;
        ``0 < iprint < 99`` print also f and ``|proj g|`` every iprint iterations;
        ``iprint = 99``   print details of every iteration except n-vectors;
        ``iprint = 100``  print also the changes of active set and final x;
        ``iprint > 100``  print details of every iteration including x and g.
    maxls : int, optional
        Maximum number of line search steps (per iteration). Default is 20.
    finite_diff_rel_step : None or array_like, optional
        If `jac in ['2-point', '3-point', 'cs']` the relative step size to
        use for numerical approximation of the jacobian. The absolute step
        size is computed as ``h = rel_step * sign(x) * max(1, abs(x))``,
        possibly adjusted to fit into the bounds. For ``method='3-point'``
        the sign of `h` is ignored. If None (default) then step is selected
        automatically.
    alpha_linesearch: float
        Parameters for linesearch. The default is 1e-4.
    beta_linesearch: float
        Parameters for linesearch. The default is 0.9.
    max_steplength: float
        Maximum steplength allowed. The default is 1e8.
    xtol_minpack: float
        Tolerance used by minpack2. The default is 1e-5.
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
    max_steplength_user = max_steplength

    # applying the bounds to the initial guess x0
    n = x0.size
    if x0.dtype != np.float64:
        x = x0.astype(np.float64, copy=True)
        x = np.clip(x, lb, ub)
    else:
        x = np.clip(x0, lb, ub)

    # Deque = similar to list but with faster operations to remove and add
    # values to extremities
    X: Deque[NDArrayFloat] = deque()
    G: Deque[NDArrayFloat] = deque()

    # search direction for the minimization problem
    W = np.zeros([n, 1])
    M = np.zeros([1, 1])
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

    task_str = "Nothing"

    while (
        np.max(np.abs(np.clip(x - grad, lb, ub) - x)) > gtol and n_iterations < max_iter
    ):
        oljac0 = f0
        x.copy()
        grad.copy()

        #
        dictCP = compute_Cauchy_point(x, grad, lb, ub, W, M, theta)

        # find the
        dictMinMod = minimize_model(
            x, dictCP["xc"], dictCP["c"], grad, lb, ub, W, M, theta
        )

        # search direction for the minimization problem
        d = dictMinMod["xbar"] - x

        # print(f"d = {d}")

        # TODO: delete this ?
        # These two portions of the workspace are described in the mainlb
        # subroutine in lbfgsb.f. See line 363.
        # s = wa[0 : m * n].reshape(m, n)
        # y = wa[m * n : 2 * m * n].reshape(m, n)

        # Get the number of past gradient and x updates to use for hessian approx
        min(n_iterations, maxcor)

        # hess_inv = LbfgsInvHessProduct(s[:n_corrs], y[:n_corrs])
        # print(np.array(S).shape)
        # print(np.array(Y).shape)
        if n_iterations != 0:
            hess_inv = LbfgsInvHessProduct(
                np.diff(np.array(X), axis=0), np.diff(np.array(G), axis=0)
            )

            #### END modify #### ???
            -hess_inv.dot(x) - x
            # print(f"d2 = {d2}")

            # d = d2

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
            alpha_linesearch,
            beta_linesearch,
            xtol_minpack,
            maxls,
        )

        if steplength is None:
            if len(X) == 0:
                # Hessian already rebooted: abort.
                task_str = "Error: can not compute new steplength : abort"
                f, grad = sf.fun_and_grad(x)
                break
            else:
                # Reboot BFGS-Hessian:
                X.clear()
                G.clear()
                W = np.zeros([n, 1])
                M = np.zeros([1, 1])
                theta = 1
        else:
            x += steplength * d
            f0, grad = sf.fun_and_grad(x)

            W, M, theta = get_lbfgs_matrices(
                x.copy(),  # copy otherwise x might be changed in X when updated
                grad,
                X,
                G,
                maxcor,
                W.copy(),
                M.copy(),
                copy.copy(theta),
                False,
                eps_SY,
            )

            print(
                "Iteration #%d (max: %d): ||x||=%.3e, f(x)=%.3e, ||jac(x)||=%.3e, "
                "cdt_arret=%.3e (eps=%.3e)"
                % (
                    n_iterations,
                    max_iter,
                    np.linalg.norm(x, np.inf),
                    f0,
                    np.linalg.norm(grad, np.inf),
                    np.max(np.abs(np.clip(x - grad, lb, ub) - x)),
                    gtol,
                )
            )
            if (oljac0 - f0) / max(abs(oljac0), abs(f0), 1) < ftol:
                task_str = "CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH"
                break
            n_iterations += 1

    if n_iterations == max_iter:
        task_str = "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT"

    # Add test for number of function reached
    # 'STOP: TOTAL NO. of ITERATIONS REACHED LIMIT'

    # Add test for max gradient
    warnflag = 0

    return OptimizeResult(
        fun=f0,
        jac=grad,
        nfev=sf.nfev,
        njev=sf.ngev,
        nit=n_iterations,
        status=warnflag,
        message=task_str,
        x=x,
        success=(warnflag == 0),
        hess_inv=LbfgsInvHessProduct(
            np.diff(np.array(X), axis=0), np.diff(np.array(G), axis=0)
        ),
    )
