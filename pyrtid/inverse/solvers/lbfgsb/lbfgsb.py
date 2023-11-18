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
from typing import Callable, Deque, Optional, Tuple, Union

import numpy as np
from scipy.optimize._lbfgsb_py import LbfgsInvHessProduct  # noqa : F401
from scipy.optimize._optimize import OptimizeResult, _prepare_scalar_function

from pyrtid.inverse.solvers.lbfgsb.base import (
    clip2bounds,
    count_var_at_bounds,
    display_iter,
    display_results,
    display_start,
    get_bounds,
    projgr,
)
from pyrtid.inverse.solvers.lbfgsb.bfgsmats import LBFGSmat, update_lbfgs_matrices
from pyrtid.inverse.solvers.lbfgsb.cauchy import get_cauchy_point
from pyrtid.inverse.solvers.lbfgsb.linesearch import line_search
from pyrtid.inverse.solvers.lbfgsb.subspacemin import (
    direct_primal_subspace_minimization,
    freev,
)
from pyrtid.utils import NDArrayFloat


def minimize_lbfgsb(
    *,
    x0: NDArrayFloat,
    fun: Optional[Callable[[NDArrayFloat, ...], float]] = None,
    args: Tuple = (),
    jac: Optional[Union[Callable[[NDArrayFloat, ...], NDArrayFloat], str, bool]] = None,
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
    ftol_linesearch: float = 1e-3,
    gtol_linesearch: float = 0.9,
    xtol_linesearch: float = 1e-1,
    eps_SY: float = 2.2e-16,
) -> OptimizeResult:
    r"""
    Solves bound constrained optimization problems by using the compact formula
    of the limited memory BFGS updates.

    fun :  Optional[Callable[[NDArrayFloat, Tuple[Any]], float]],
        The objective function to be minimized.

            ``fun(x, *args) -> float``

        where ``x`` is a 1-D array with shape (n,) and ``args``
        is a tuple of the fixed parameters needed to completely
        specify the function. Mandatory if `fun_and_jax` is not specified. The default
        is None.
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

    # Some display about the problem at hand. The display depends on the value of iprint
    display_start(
        np.finfo(float).eps, n, maxcor, count_var_at_bounds(x, lb, ub), iprint
    )

    # Deque = similar to list but with faster operations to remove and add
    # values to extremities
    X: Deque[NDArrayFloat] = deque()
    G: Deque[NDArrayFloat] = deque()

    # Saved parameters in linesearch
    isave = np.zeros((2,), np.intc)
    dsave = np.zeros((13,), float)

    # Initialize lbfgsb matrices ("Wa", "Ws", "Wy", "Sy", "Ss", "Wn", "Wt", "Snd")
    LBFGSmat(x.size, maxcor)

    # search direction for the minimization problem
    W: NDArrayFloat = np.zeros([n, 1])
    M: NDArrayFloat = np.zeros([1, 1])
    invMfactors: Tuple[NDArrayFloat, NDArrayFloat] = (
        np.zeros([1, 1]),
        np.zeros([1, 1]),
    )
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
            invMfactors,
            theta,
            len(X),
            maxcor,
            iprint,
            n_iterations,
        )

        # Get the free variables for the GCP
        free_vars, Z, A = freev(dictCP["xc"], lb, ub, iprint, n_iterations, free_vars)

        # subspace minimization: find the search direction for the minimization problem
        xbar: NDArrayFloat = direct_primal_subspace_minimization(
            X,
            G,
            x,
            dictCP["xc"],
            free_vars,
            Z,
            A,
            dictCP["c"],
            grad,
            lb,
            ub,
            W,
            invMfactors,
            theta,
            n_iterations,
        )
        d = xbar - x

        steplength = line_search(
            x,
            f0,
            grad,
            d,
            lb,
            ub,
            n_iterations,
            max_steplength_user,
            sf.fun_and_grad,
            ftol_linesearch,
            gtol_linesearch,
            xtol_linesearch,
            maxls,
            iprint,
            isave,
            dsave,
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

            # new evaluation
            f0, grad = sf.fun_and_grad(x)

            # perform a potential update of the objective function definition and
            # upgrade the gradient and the past sequence of gradients accordingly
            if update_fun_def is not None:
                f0, grad, G = update_fun_def(x, f0, grad, X, G)

            W, M, invMfactors, theta = update_lbfgs_matrices(
                x.copy(),  # copy otherwise x might be changed in X when updated
                grad,
                X,
                G,
                maxcor,
                W.copy(),
                M.copy(),
                invMfactors,
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
