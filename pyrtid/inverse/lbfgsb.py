from collections import deque
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from scipy.optimize import minpack2
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

    .. seealso::

       [1] R. H. Byrd, P. Lu, J. Nocedal and C. Zhu, ``A limited
       memory algorithm for bound constrained optimization'',
       SIAM J. Scientific Computing 16 (1995), no. 5, pp. 1190--1208.

       [2] C. Zhu, R.H. Byrd, P. Lu, J. Nocedal, ``L-BFGS-B: FORTRAN
       Subroutines for Large Scale Bound Constrained Optimization''
       Tech. Report, NAM-11, EECS Department, Northwestern University,
       1994.
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

            .. seealso::

       [1] R. H. Byrd, P. Lu, J. Nocedal and C. Zhu, ``A limited
       memory algorithm for bound constrained optimization'',
       SIAM J. Scientific Computing 16 (1995), no. 5, pp. 1190--1208.

       [2] C. Zhu, R.H. Byrd, P. Lu, J. Nocedal, ``L-BFGS-B: FORTRAN
       Subroutines for Large Scale Bound Constrained Optimization''
       Tech. Report, NAM-11, EECS Department, Northwestern University,
       1994.
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

    .. seealso::

       [1] R. H. Byrd, P. Lu, J. Nocedal and C. Zhu, ``A limited
       memory algorithm for bound constrained optimization'',
       SIAM J. Scientific Computing 16 (1995), no. 5, pp. 1190--1208.

       [2] C. Zhu, R.H. Byrd, P. Lu, J. Nocedal, ``L-BFGS-B: FORTRAN
       Subroutines for Large Scale Bound Constrained Optimization''
       Tech. Report, NAM-11, EECS Department, Northwestern University,
       1994.
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

    .. seealso::

       [minpack] scipy.optimize.minpack2.dcsrch

       [1] R. H. Byrd, P. Lu, J. Nocedal and C. Zhu, ``A limited
       memory algorithm for bound constrained optimization'',
       SIAM J. Scientific Computing 16 (1995), no. 5, pp. 1190--1208.

       [2] C. Zhu, R.H. Byrd, P. Lu, J. Nocedal, ``L-BFGS-B: FORTRAN
       Subroutines for Large Scale Bound Constrained Optimization''
       Tech. Report, NAM-11, EECS Department, Northwestern University,
       1994.
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


def update_SY(sk, yk, S, Y, m, W, M, thet, eps=2.2e-16):
    """
        Update lists S and Y, and form the L-BFGS Hessian approximation thet, W and M.

    :param sk: correction in x = new_x - old_x
    :type sk: np.array

    :param yk: correction in gradient = f'(new_x) - f'(old_x)
    :type yk: np.array

    :param S, Y: lists defining the L-BFGS matrices, updated during process (IN/OUT)
    :type S, Y: list

    :param m: Maximum size of lists S and Y: keep in memory only m previous iterations
    :type m: integer

    :param W, M: L-BFGS matrices
    :type W, M: np.array

    :param thet: L-BFGS float parameter
    :type thet: float

    :param eps: Positive stability parameter for accepting current step for updating
    matrices.
    :type eps: float >0

    :return: updated [W, M, thet]
    :rtype: tuple

    .. seealso::

       [1] R. H. Byrd, P. Lu, J. Nocedal and C. Zhu, ``A limited
       memory algorithm for bound constrained optimization'',
       SIAM J. Scientific Computing 16 (1995), no. 5, pp. 1190--1208.

       [2] C. Zhu, R.H. Byrd, P. Lu, J. Nocedal, ``L-BFGS-B: FORTRAN
       Subroutines for Large Scale Bound Constrained Optimization''
       Tech. Report, NAM-11, EECS Department, Northwestern University,
       1994.
    """
    sTy = sk.dot(yk)
    yTy = yk.dot(yk)
    if sTy > eps * yTy:
        S.append(sk)
        Y.append(yk)
        if len(S) > m:
            S.popleft()
            Y.popleft()
        Sarray = np.asarray(S).T
        Yarray = np.asarray(Y).T
        STS = np.transpose(Sarray).dot(Sarray)
        L = np.transpose(Sarray).dot(Yarray)
        D = np.diag(-np.diag(L))
        L = np.tril(L, -1)

        thet = yTy / sTy
        W = np.hstack([Yarray, thet * Sarray])
        M = np.linalg.inv(np.hstack([np.vstack([D, L]), np.vstack([L.T, thet * STS])]))

    return [W, M, thet]


def L_BFGS_B(
    *,
    x0: NDArrayFloat,
    fun: Callable[[NDArrayFloat], float],
    args: Tuple = (),
    jac=Optional[Callable[[NDArrayFloat], float]],
    bounds: Optional[NDArrayFloat] = None,
    m: int = 10,
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
    max_iter_linesearch: int = 30,
    eps_SY: float = 2.2e-16,
):
    """
       Solves bound constrained optimization problems by using the compact formula
       of the limited memory BFGS updates.

    :param x0: initial guess
    :type sk: np.array

    :param f: cost function to optimize f(x)
    :type f: function returning float

    :param jac: gradient of cost function to optimize f'(x)
    :type jac: function returning np.array

    :param l: the lower bound of x
    :type l: np.array

    :param u: the upper bound of x
    :type u: np.array

    :param m: Maximum size of lists for L-BFGS Hessian approximation
    :type m: integer

    :param gtol: Tolerance on projected gradient: programs converges when
                P(x-grad, l, u)<gtol.
    :type gtol: float

    :param ftol: Tolerance on function change: programs ends when
    (f_k-f_{k+1})/max(|f_k|,|f_{k+1}|,1) < ftol
    :type ftol: float

    :param alpha_linesearch, beta_linesearch: Parameters for linesearch.
                                              See ``alpha`` and ``beta``
                                              in :func:`line_search`
    :type alpha_linesearch, beta_linesearch: float

    :param max_steplength: Maximum steplength allowed. See ``max_steplength``
    in :func:`max_allowed_steplength`
    :type max_steplength: float

    :param xtol_minpack: Tolerance used by minpack2. See ``xtol_minpack``
    in :func:`line_search`
    :type xtol_minpack: float

    :param max_iter_linesearch: Maximum number of trials for linesearch.
                                See ``max_iter_linesearch`` in :func:`line_search`
    :type max_iter_linesearch: integer

    :param eps_SY: Parameter used for updating the L-BFGS matrices. See ``eps``
    in :func:`update_SY`
    :type eps_SY: float

    :return: dict containing:
            - 'x': optimal point
            - 'f': optimal value at x
            - 'jac': gradient f'(x)
    :rtype: dict


    ..todo Check matrices update and different safeguards may be missing

    .. seealso::
       Function tested on Rosenbrock and Beale function with different starting points.
       All tests passed.

       [1] R. H. Byrd, P. Lu, J. Nocedal and C. Zhu, ``A limited
       memory algorithm for bound constrained optimization'',
       SIAM J. Scientific Computing 16 (1995), no. 5, pp. 1190--1208.

       [2] C. Zhu, R.H. Byrd, P. Lu, J. Nocedal, ``L-BFGS-B: FORTRAN
       Subroutines for Large Scale Bound Constrained Optimization''
       Tech. Report, NAM-11, EECS Department, Northwestern University,
       1994.
    """
    lb = bounds[:, 0]
    ub = bounds[:, 1]

    max_steplength_user = max_steplength

    n = x0.size
    if x0.dtype != np.float64:
        x = x0.astype(np.float64, copy=True)
        x = np.clip(x, lb, ub)
    else:
        x = np.clip(x0, lb, ub)
    S = deque()
    Y = deque()
    W = np.zeros([n, 1])
    M = np.zeros([1, 1])
    theta = 1

    sf = _prepare_scalar_function(
        fun,
        x0,
        jac=jac,
        args=args,
        epsilon=eps,
        bounds=bounds,
        finite_diff_rel_step=finite_diff_rel_step,
    )
    func_and_grad = sf.fun_and_grad

    f0, grad = func_and_grad(x)
    n_iterations = 0

    task_str = "Nothing"

    while (
        np.max(np.abs(np.clip(x - grad, lb, ub) - x)) > gtol and n_iterations < max_iter
    ):
        oljac0 = f0
        oldx = x.copy()
        oldg = grad.copy()
        dictCP = compute_Cauchy_point(x, grad, lb, ub, W, M, theta)
        dictMinMod = minimize_model(
            x, dictCP["xc"], dictCP["c"], grad, lb, ub, W, M, theta
        )

        d = dictMinMod["xbar"] - x

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
            max_iter_linesearch,
        )

        if steplength is None:
            if len(S) == 0:
                # Hessian already rebooted: abort.
                task_str = "Error: can not compute new steplength : abort"
                f, grad = sf.fun_and_grad(x)
                return {"x": x, "f": f, "jac": grad}
            else:
                # Reboot BFGS-Hessian:
                S.clear()
                Y.clear()
                W = np.zeros([n, 1])
                M = np.zeros([1, 1])
                theta = 1
        else:
            x += steplength * d
            f0, grad = func_and_grad(x)

            # On met à jour l'inverse du gradient
            [W, M, theta] = update_SY(
                x - oldx, grad - oldg, S, Y, m, W, M, theta, eps_SY
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
        hess_inv=None,
    )
