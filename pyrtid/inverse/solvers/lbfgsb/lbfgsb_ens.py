import copy
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, List, Optional, Tuple

import numpy as np
from scipy.optimize._lbfgsb_py import LbfgsInvHessProduct  # noqa : F401
from scipy.optimize._optimize import OptimizeResult

from pyrtid.inverse.solvers.lbfgsb.base import (
    clip2bounds,
    count_var_at_bounds,
    display_iter,
    display_start,
    get_bounds,
    projgr,
    projgr_ens,
)
from pyrtid.inverse.solvers.lbfgsb.bfgsmats import (
    update_lbfgs_matrices,
)
from pyrtid.inverse.solvers.lbfgsb.cauchy import get_cauchy_point
from pyrtid.inverse.solvers.lbfgsb.linesearch import line_search
from pyrtid.inverse.solvers.lbfgsb.subspacemin import (
    direct_primal_subspace_minimization,
    freev,
)
from pyrtid.utils import NDArrayBool, NDArrayFloat


@dataclass
class ScalarFunction:
    fun_and_jac: Callable[[NDArrayFloat], Tuple[float, NDArrayFloat]]
    args: Tuple
    nfev: int = 0
    ngev: int = 0

    def fun_and_grad(self, x: NDArrayFloat) -> Tuple[float, NDArrayFloat]:
        self.nfev += 1
        self.ngev += 1
        return self.fun_and_jac(x, *self.args)


@dataclass
class InternalParams:
    ftol: float
    gtol: float
    n: int
    ne: int
    n_iterations: int
    max_steplength_user: float
    ftol_linesearch: float
    gtol_linesearch: float
    xtol_linesearch: float
    maxls: int
    eps_SY: float
    maxcor: int
    iprint: int


def minimize_ensemble_lbfgsb(
    *,
    x0: NDArrayFloat,
    fun_and_jac: Callable[[NDArrayFloat, ...], Tuple[float, NDArrayFloat]],
    args: Tuple = (),
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

    fun_and_jac: Callable[[NDArrayFloat, ...], Tuple[float, NDArrayFloat]]
        The objective function to be minimized and associated jacobian.

            ``fun(x, *args) -> float``

        where ``x`` is a 1-D array with shape (n,) and ``args``
        is a tuple of the fixed parameters needed to completely
        specify the function. Mandatory if `fun_and_jax` is not specified. The default
        is None.
        TODO:
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
    x0 : NDArrayFloat
        Ensemble of initial guesses. Array of real elements of shape
        of shape (:math:`N_{n}`, :math:`N_{e}`)
        with :math:`N_{n}` the number of adjusted independent variables and
        :math:`N_{e}` the number of columns (members in the ensemble, aka
        realizations).
    args : tuple, optional
        Extra arguments passed to the objective function and its
        derivatives (`fun`, `jac` and `hess` functions).
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
    x = clip2bounds(x0, lb, ub)

    # instance holding the parameters of the optimization
    iparams = InternalParams(
        ftol=ftol,
        gtol=gtol,
        n=x0.shape[0],  # type: ignore
        ne=x0.shape[1],  # type: ignore
        n_iterations=0,
        max_steplength_user=max_steplength_user,
        ftol_linesearch=ftol_linesearch,
        gtol_linesearch=gtol_linesearch,
        xtol_linesearch=xtol_linesearch,
        maxls=maxls,
        eps_SY=eps_SY,
        maxcor=maxcor,
        iprint=iprint,
    )

    # Some display about the problem at hand. The display depends on the value of iprint
    display_start(
        np.finfo(float).eps,
        iparams.n,
        iparams.maxcor,
        count_var_at_bounds(x, lb, ub),
        iparams.iprint,
    )

    # Deque = similar to list but with faster operations to remove and add
    # values to extremities. This is more expensive
    deque()
    deque()
    X: Deque[NDArrayFloat] = deque()
    G: Deque[NDArrayFloat] = deque()

    # search direction for the minimization problem
    W: NDArrayFloat = np.zeros([iparams.n, 1])
    M: NDArrayFloat = np.zeros([1, 1])
    invMlt: NDArrayFloat = np.zeros([1, 1])
    theta = 1

    # wrapper storing the calls to f and g and handling finite difference approximation
    sf = ScalarFunction(fun_and_jac, args)

    # Parallel estimation of f0 and gradx0.
    # TODO: in parallel
    flist: List[float] = []
    glist: List[NDArrayFloat] = []
    for i, _x in enumerate(x.T):
        f0, grad = sf.fun_and_grad(_x)
        flist.append(f0)
        glist.append(grad)

    grad = np.array(glist).T

    n_iterations = 0
    task_str = "START"
    is_sucess = False
    warnflag = 2

    # For now the free variables at the cauchy points is an empty set
    np.array([], dtype=np.int_)

    # Check the infinity norm of the projected gradient
    # Check the smalest projected gradient infinity norm
    sbgnrm: NDArrayFloat = projgr_ens(x, grad, lb, ub)

    display_iter(n_iterations, np.max(sbgnrm), max(flist), iprint)

    # Create an array of status to indicate which realization has reached a stop
    # criteria. For now none has converged.
    # Initialized with the projected gradient
    has_converged = sbgnrm <= iparams.gtol

    print(has_converged)

    # Store first res to X and G and update the BFGS matrices with the ensemble.
    # Note: we do not use the gradient of members for which the projection
    # criterion (pgtol) is already met
    # First get the index of the first valid member
    fst_index: int = np.nonzero(~has_converged)[0][0]
    print(f"fist_index: {fst_index}")

    # TODO: ici on a déja Ne gradient. On peut donc mettre à jour les matrices
    # directement.
    # For the first Hessian approximation, we find the minimum objective function
    minf_index = np.argsort(np.array(flist))[0]

    # For this
    X.append(x[:, fst_index])
    G.append(glist[fst_index])

    print(minf_index)

    W, M, invMlt, theta = update_lbfgs_matrices(
        x[:, fst_index + 1 :],
        np.array(glist).T[:, fst_index + 1 :],
        X,
        G,
        maxcor,
        W.copy(),
        M.copy(),
        invMlt,
        copy.copy(theta),
        False,
        iparams.eps_SY,
    )

    # Note that interruptions due to maxfun are postponed
    # until the completion of the current minimization iteration.
    while not has_converged.all() and n_iterations < max_iter and sf.nfev < maxfun:
        if iprint > 99:
            print(f"\nITERATION {n_iterations + 1}\n")

        x.copy()
        grad.copy()

        for rindex in range(iparams.ne):
            # do not update members that have converged
            if has_converged[rindex]:
                continue
            update_member(
                x[:, rindex],
                grad[:, rindex],
                rindex,
                flist,
                lb,
                ub,
                sf,
                W,
                M,
                len(X),
                invMlt,
                theta,
                iparams,
                has_converged,
            )
            # perform a potential update of the objective function definition and
            # upgrade the gradient and the past sequence of gradients accordingly

        # Update the matrices only with
        W, M, invMlt, theta = update_lbfgs_matrices(
            x,  # copy otherwise x might be changed in X when updated
            # old_x,
            grad,
            # old_grad,
            X,
            G,
            maxcor,
            W.copy(),
            M.copy(),
            invMlt,
            copy.copy(theta),
            False,
            eps_SY,
        )

        print(has_converged)

        # TODO: handle the callbacks
        # # callback is a user defined mechanism to stop optimization
        # # if callback returns True, then it stops.
        # if callback is not None:
        #     if callback(
        #         np.copy(x),
        #         OptimizeResult(
        #             fun=f0,
        #             jac=grad,
        #             nfev=sf.nfev,
        #             njev=sf.ngev,
        #             nit=n_iterations,
        #             status=warnflag,
        #             message=task_str,
        #             x=x,
        #             success=is_sucess,
        #             hess_inv=LbfgsInvHessProduct(
        #                 np.diff(np.array(X), axis=0), np.diff(np.array(G), axis=0)
        #             ),
        #         ),
        #     ):
        #         task_str = "STOP: USER CALLBACK"
        #         is_sucess = True
        #         break

        # Need a wrapper for this
        # End of the for loop over the _x

        # Result display
        # display_results(
        #     iprint, n_iterations, max_iter, x, grad, lb, ub, f0, gtol, False
        # )

        # display_iter(n_iterations + 1, sbgnrm, f0, iprint)

        # Update the objective function with a new regularization
        # if update_fun_def is not None:
        #     f0, grad, G = update_fun_def(_x, f0, _grad, X, G)

        n_iterations += 1

    # Final display
    # display_results(iprint, n_iterations, max_iter, x, grad, lb, ub, f0, gtol, True)

    print(projgr_ens(x, grad, lb, ub))

    if (projgr_ens(x, grad, lb, ub) <= gtol).any():
        task_str = "CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL"
        is_sucess = True
        warnflag = 1
    if n_iterations >= max_iter:
        task_str = "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT"
        is_sucess = True
        warnflag = 1
    elif sf.nfev >= maxfun:
        task_str = "STOP: TOTAL NO. of f AND g EVALUATIONS EXCEEDS LIMIT"
        is_sucess = True
        warnflag = 1

    print(flist)

    # error: b'ERROR: STPMAX .LT. STPMIN'
    return OptimizeResult(
        fun=min(flist),
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


def update_member(
    x: NDArrayFloat,
    grad: NDArrayFloat,
    rindex: int,
    flist: List[float],
    lb: NDArrayFloat,
    ub: NDArrayFloat,
    sf: ScalarFunction,
    W: NDArrayFloat,
    M: NDArrayFloat,
    ncor: int,
    invMlt: NDArrayFloat,
    theta: float,
    iparams: InternalParams,
    has_converged: NDArrayBool,
) -> None:
    # Step 1) find cauchy point
    # TODO: replace dictCP by a class
    dictCP = get_cauchy_point(
        x,
        grad,
        lb,
        ub,
        W,
        M,
        invMlt,
        theta,
        ncor,
        iparams.maxcor,
        iparams.iprint,
        iparams.n_iterations,
    )

    # Step 2) Get the free variables for the GCP

    free_vars, Z, A = freev(dictCP["xc"], lb, ub, iparams.iprint, iparams.n_iterations)

    # if n_iterations != 0 and dictCP["free_vars"] != 0:
    # Factorization of the matrix K used in the subspace minimization
    # TODO: there is something I don't get here...
    # K: NDArrayFloat = formk(X, G, Z, A, theta)
    K = None

    # Step 3) subspace minimization: find the search direction for the minimization
    # problem
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

    # Step 4)
    steplength = line_search(
        x,
        flist[rindex],
        grad,
        d,
        lb,
        ub,
        iparams.n_iterations,
        iparams.max_steplength_user,
        sf.fun_and_grad,
        iparams.ftol_linesearch,
        iparams.gtol_linesearch,
        iparams.xtol_linesearch,
        iparams.maxls,
        iparams.iprint,
    )

    if steplength is None:
        print(flist[rindex])
        print(x)
        print(grad)
        # If the linesearch does not succeed for one member,
        # then we will try again at the next iteration with the updated
        # Hessian. Since x has not been update, no need to recompute the
        # gradient nor the objective function.
        return  # -> end the loop over the realizations

    # x update in place
    x += steplength * d

    oldf0 = flist[rindex].copy()
    # Update the objective function and the gradient
    f0, grad = sf.fun_and_grad(x)

    flist[rindex] = f0

    # Check the convergence of the ensemble member
    if projgr(x, grad, lb, ub) <= iparams.gtol:
        has_converged[rindex] = True
    if (oldf0 - f0) / max(abs(oldf0), abs(f0), 1) < iparams.ftol:
        has_converged[rindex] = True
