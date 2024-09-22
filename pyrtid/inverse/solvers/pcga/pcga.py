"""
Implement the Principal Component Geostatistical Approach for large-scale inversion.

The original code has been written by Jonghyun Harry Lee.

See: https://github.com/jonghyunharrylee/pyPCGA
"""

import logging
import multiprocessing
from dataclasses import dataclass, field
from math import isnan, sqrt
from time import time
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import scipy as sp
from scipy._lib._util import check_random_state  # To handle random_state
from scipy.sparse.linalg import LinearOperator, eigsh, gmres, minres, svds

from pyrtid.inverse.regularization import (
    ConstantDriftMatrix,
    DriftMatrix,
    EigenFactorizedCovarianceMatrix,
)
from pyrtid.utils import NDArrayFloat, StrEnum

VERY_LARGE_NUMBER = 1.0e20


class PostCovEstimation(StrEnum):
    DIAGONAL = "diagonal"
    DIRECT = "direct"


class Residual:
    def __init__(self) -> None:
        self.res: List[NDArrayFloat] = []

    def __call__(self, rk: NDArrayFloat) -> None:
        self.res.append(rk)

    def itercount(self) -> int:
        return len(self.res)

    def clear(self) -> None:
        self.res = []


@dataclass
class InternalState:
    """Class to keep track of internal state."""

    # keep track of some values (best, init)
    s_best: NDArrayFloat
    beta_best: NDArrayFloat = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )
    simul_obs_best: NDArrayFloat = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )
    iter_best: int = 0
    simul_obs_init: NDArrayFloat = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )
    objvals: List[float] = field(default_factory=lambda: [])
    Q2_all: NDArrayFloat = field(default_factory=lambda: np.array([], dtype=np.float64))
    cR_all: NDArrayFloat = field(default_factory=lambda: np.array([], dtype=np.float64))

    Q2_best: NDArrayFloat = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )
    cR_best: NDArrayFloat = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )
    i_best: int = 0
    status: str = "ILDE."
    is_success: bool = False

    @property
    def obj_best(self) -> float:
        """
        Return the best objective function obtained in the optimization.

        The first objective function is ignored because beta is minimized.
        """
        if len(self.objvals) == 0:
            return np.inf
        if len(self.objvals) == 1:
            return 1e20  # very high value to avoid convergence
        return float(np.min(self.objvals))

    # TODO: I am not sure about this (21/09/2024)
    @property
    def Q2_cur(self) -> float:
        return self.Q2_all[:, -2:-1]

    @property
    def cR_cur(self) -> float:
        return self.cR_all[:, -2:-1]


class PCGA:
    """
    Solve inverse problem with PCGA (approx to quasi-linear method)

    every values are represented as 2D np array
    """

    def __init__(
        self,
        s_init: NDArrayFloat,
        obs: NDArrayFloat,
        cov_obs: Union[float, NDArrayFloat],
        forward_model: Callable,
        Q: EigenFactorizedCovarianceMatrix,
        drift: Optional[DriftMatrix] = None,
        prior_s_var: Optional[Union[float, NDArrayFloat]] = None,
        callback: Optional[Callable] = None,
        is_line_search: bool = False,
        is_lm: bool = False,
        is_direct_solve: bool = False,
        is_use_preconditioner: bool = False,
        random_state: Optional[
            Union[int, np.random.Generator, np.random.RandomState]
        ] = np.random.default_rng(),
        post_cov_estimation: Optional[PostCovEstimation] = None,
        is_objfun_exact: bool = False,  # former objeval
        max_it_lm: int = multiprocessing.cpu_count(),
        alphamax_lm: float = 10.0**3.0,  # does it sound ok?
        lm_smin: Optional[float] = None,
        lm_smax: Optional[float] = None,
        max_it_ls: int = 20,
        maxiter: int = 10,
        ftol: float = 1e-5,
        ftarget: Optional[float] = None,
        restol: float = 1e-2,
        is_post_cov: bool = False,
        logger: Optional[logging.Logger] = None,
        is_save_jac: bool = False,
        # PCGA parameters (perturbation size)
        eps=1.0e-8,
    ) -> None:
        r"""
        Initialize the instance.

        Parameters
        ----------
        s_init : NDArrayFloat
            1D array of initial control parameters, i.e., initial solution for
            Gauss-Newton method. In theory, the choice of
            s_init does not affect the estimation while total number of
            iterations/number of forward model runs depend on `s_init`.
        obs : numpy.ndarray, optional
            1D array of (noisy) measurements used for inversion.
        cov_obs : NDArrayFloat
            Covariance matrix of observed data measurement errors with dimensions
            (:math:`N_{obs}`, :math:`N_{obs}`). Also denoted :math:`R`.
            If a 1D array is passed, it represents a diagonal covariance matrix.
            If a float is passed, it means the noise is the same for all
            measurements.
        forward_model : Callable
            Wrapper for forward model obs = f(s). See a template python file in each
            example for more information.
        Q : EigenFactorizedCovarianceMatrix
            _description_
        drift : Optional[DriftMatrix], optional
            _description_, by default None
        prior_s_var : Optional[Union[float, NDArrayFloat]], optional
            _description_, by default None
        callback : Optional[Callable], optional
            _description_, by default None
        is_line_search : bool, optional
            _description_, by default False
        is_lm : bool, optional
            _description_, by default False
        is_direct_solve : bool, optional
            _description_, by default False
        is_use_preconditioner : bool, optional
            _description_, by default False
        random_state : Optional[ Union[int, np.random.Generator, np.random.RandomState]]
            Pseudorandom number generator state used to generate resamples.
            If `random_state` is ``None`` (or `np.random`), the
            `numpy.random.RandomState` singleton is used.
            If `random_state` is an int, a new ``RandomState`` instance is used,
            seeded with `random_state`.
            If `random_state` is already a ``Generator`` or ``RandomState``
            instance then that instance is used.
        post_cov_estimation : Optional[PostCovEstimation], optional
            _description_, by default None
        is_objfun_exact : bool, optional
            _description_, by default False
        alphamax_lm : float, optional
            _description_, by default 10.0**3.0
        lm_smax : Optional[float], optional
            _description_, by default None
        max_it_ls : int, optional
            _description_, by default 20
        maxiter : int, optional
            _description_, by default 10
        ftarget: Optional[Union[float, Callable]] = None, optional
            Target objective function (stop criterion) .
            The iteration stops when ``f^{k+1} <= fmin``. If None, the stop criterion
            is ignored. The default is None.
        ftol : float, optional
            Objective function minimum change (stop criterion). The iteration stops
            when ``(f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= ftol``.
            Typical values for `ftol` on a computer with 15 digits of accuracy in double
            precision are as follows: `ftol` = 5e-3 for low accuracy; `ftol` = 5e-8
            for moderate accuracy; `ftol` = 5e-14 for extremely high accuracy.
            If `ftol` = 0, the test will stop the algorithm only if the objective
            function remains unchanged after one iteration. The default is 1e-5.
        restol : float, optional
            Mininmum change in the update value. Below this threshold, changes are
            considered unisgnificant and inversion is stopped.
            The change is computed as the euclidean norm of the difference between the
            current updated values and the values at the previous iteration scaled by
            the euclidean norm of the values at the previous iteration:
            .. math::
                \mathrm{change} = \dfrac{\left\lVert \mathbf{s}_{\ell+1} -
                \mathbf{s}_{\ell} \right\rVert^{2}}{\left\lVert
                \mathbf{s}_{\ell} \right\rVert^{2}}
            by default 1e-2.
        is_post_cov : bool, optional
            _description_, by default False
        logger: Optional[Logger], optional
            Logger, by default None.
        is_save_jac : bool, optional
            _description_, by default False
        eps : _type_, optional
            _description_, by default 1.0e-8
        """
        ##### Forward Model
        # Make sure the array has a second dimension of length 1.
        self.s_init = np.array(s_init).reshape(-1, 1)
        # Observations
        self.obs = np.array(obs).reshape(-1, 1)
        self.cov_obs = np.asarray(cov_obs)
        # forward solver setting should be done externally as a blackbox
        # including the parallelization
        self.forward_model = forward_model
        self.Q: EigenFactorizedCovarianceMatrix = Q
        self.callback: Optional[Callable] = callback
        self.is_line_search: bool = is_line_search
        self.is_lm: bool = is_lm
        self.is_direct_solve: bool = is_direct_solve
        self.is_use_preconditioner: bool = is_use_preconditioner
        self.is_objfun_exact: bool = is_objfun_exact
        self.max_it_lm = max_it_lm
        self.alphamax_lm: float = alphamax_lm
        self.lm_smin: Optional[float] = lm_smin
        self.lm_smax: Optional[float] = lm_smax
        self.max_it_ls: int = max_it_ls
        self.maxiter: int = maxiter
        self.ftol: float = ftol
        self.restol: float = restol
        self.ftarget: Optional[float] = ftarget
        self.is_post_cov: bool = is_post_cov
        self.post_cov_estimation: Optional[PostCovEstimation] = post_cov_estimation
        # Switch to direct if direct solve:
        # Otherwise the preconditioner is not build while it is required
        # for the diagonal post covariance estimation
        if self.post_cov_estimation is not None and self.is_direct_solve:
            self.post_cov_estimation = PostCovEstimation.DIRECT

        # PCGA parameters (purturbation size)
        self.eps: float = eps

        # TODO: parametrize
        self.nopts_lm = 4

        # keep track of the internal state
        self.istate = InternalState(s_best=s_init)
        # Random state for v0 vector used by eigsh and svds
        self.random_state = check_random_state(random_state)

        if prior_s_var is not None:
            self.prior_s_var = prior_s_var
        else:
            self.prior_s_var = self.Q.get_diagonal()

        # Initialized as the diagonal of the covariance matrix
        self.post_diagv = self.prior_s_var

        # Define Drift (or Prior) functions
        if drift is not None:
            assert drift.s_dim == self.s_dim
            self.drift: DriftMatrix = drift
        else:
            self.drift = ConstantDriftMatrix(self.s_dim)

        # Internal state
        self.is_save_jac = is_save_jac
        if self.post_cov_estimation is not None:
            self.is_save_jac = True

        # Need the preconditionner if PostCovEstimation Diagonal
        self.is_use_preconditioner = (
            is_use_preconditioner
            or self.post_cov_estimation == PostCovEstimation.DIAGONAL
        )

        self.logger: Optional[logging.Logger] = logger

        # TODO: see if we move these internal states
        self.HX = None
        self.HZ = None
        self.Hs = None
        self.P = None
        self.Psi_U = None
        self.Psi_sigma = None

        ##### Optimization
        self.display_init_parameters()

    def loginfo(self, msg: str) -> None:
        if self.logger is not None:
            self.logger.info(msg)

    def display_init_parameters(self) -> None:
        self.loginfo("##### PCGA Inversion #####")
        self.loginfo("##### 1. Initialize forward and inversion parameters")
        self.loginfo("------------ Inversion Parameters -------------------------")
        _dict = {
            "Number of unknowns": self.s_dim,
            "Number of observations": self.d_dim,
            "Number of principal components (n_pc)": self.Q.n_pc,
            "Maximum Gauss-Newton iterations": self.maxiter,
            "Machine eps (delta = sqrt(eps))": self.eps,
            "Minimum model change (restol)": np.round(self.restol, 3),
            "Minimum obj fun change (ftol)": np.round(self.ftol, 3),
            "Target obj fun (ftarget)": (
                np.round(self.ftarget, 3) if self.ftarget is not None else None
            ),
            "Levenberg-Marquardt (is_lm)": self.is_lm,
            "Posterior covariance computation": self.post_cov_estimation,
        }
        if self.is_lm:
            _dict["Minimum LM solution (lm_smin)"] = self.lm_smin
            _dict["Maximum LM solution (lm_smax)"] = self.lm_smax
            _dict["Maximum LM iterations (lm_smax)"] = self.max_it_lm

        _dict["Line search"] = self.is_line_search

        if self.is_line_search:
            _dict["Maximum line-search iterations (max_it_ls)"] = self.max_it_ls

        # dipslay the dict content
        # first get the max length
        max_length: int = int(np.max([len(_str) for _str in _dict.keys()]))
        for k, v in _dict.items():
            self.loginfo(f"  {k: <{max_length}} : {v}")

        self.loginfo("-----------------------------------------------------------")

    @property
    def s_dim(self) -> int:
        """Return the length of the parameters vector."""
        return self.s_init.size  # type: ignore

    @property
    def d_dim(self) -> int:
        """Return the number of forecast data."""
        return self.obs.size

    @property
    def cov_obs(self) -> NDArrayFloat:
        """Get the observation errors covariance matrix."""
        return self._cov_obs

    @cov_obs.setter
    def cov_obs(self, cov: NDArrayFloat) -> None:
        """
        Set the observation errors covariance matrix.

        It must be a 2D array, or a 1D array if the covariance matrix is diagonal.
        """
        error = ValueError(
            "cov_obs must be either a 1D matrix of {self.s_dim} elements, or "
            f"a 2D matrix with dimensions ({self.d_dim}, {self.d_dim})."
        )

        if cov.size == 1:
            # Case of a float
            cov = np.ones((self.d_dim)) * cov
        else:
            # Case of 1D or 2D array with more than one value
            if cov.ndim > 2:
                raise error
            if cov.shape[0] != self.obs.size:  # type: ignore
                raise error
            if cov.ndim == 2:
                if cov.shape[0] != cov.shape[1]:  # type: ignore
                    raise error
        # From iterative_ensemble_smoother code
        # Only compute the covariance factorization once
        # If it's a full matrix, we gain speedup by only computing cholesky once
        # Note that we store the upper triangle.
        # If it's a diagonal, we gain speedup by never having to compute cholesky
        if cov.ndim == 2:
            self.cov_obs_cholesky: NDArrayFloat = sp.linalg.cholesky(cov, lower=False)
        else:
            self.cov_obs_cholesky = np.sqrt(cov)  # type: ignore

        # For now only 1D arrays are supported
        self._cov_obs: NDArrayFloat = cov

    def cov_obs_solve(self, v: NDArrayFloat) -> NDArrayFloat:
        """Return cov_obs^{-1} @ v."""
        if self.cov_obs.ndim == 2:
            # The false means we use the upper triangle which is store in
            # self.cov_obs
            return sp.linalg.cho_solve((self.cov_obs_solve, False), v)
        # Case 1D, the inverse of cov_obs (R matrix) is a diagonal matrix
        return (1.0 / self.cov_obs) * v

    @property
    def prior_s_var(self) -> NDArrayFloat:
        """Get the a priori variance of the control variables."""
        return self._prior_s_var

    @prior_s_var.setter
    def prior_s_var(self, values: Union[float, NDArrayFloat]) -> None:
        """Set the a priori variance of the control variables."""
        _values = np.asarray(values)
        if _values.size == 1:
            self._prior_s_var: NDArrayFloat = np.ones(self.s_dim) * _values.ravel()[0]
        elif _values.size == self.s_dim:
            self._prior_s_var = _values.ravel()
        else:
            raise ValueError(
                "prior_s_var must be either a float value, either a 1D "
                "array with the same number of elements as in s_init!"
            )

    def get_v0(self, size) -> Optional[NDArrayFloat]:
        if self.random_state is not None:
            return self.random_state.uniform(size=(size,))
        else:
            return None

    def jac_vect(self, x, s, simul_obs, eps, delta=None):
        """
        Jacobian times Matrix (Vectors) in Parallel
        perturbation interval delta determined following Brown and Saad [1990]
        """
        nruns = np.size(x, 1)

        # TODO: create a function perturb x (make an ensemble of perturbed values)
        # And test the function outside the loop
        deltas = np.zeros((nruns, 1), "d")

        if delta is None or isnan(delta) or delta == 0:
            for i in range(nruns):
                mag = np.dot(s.T, x[:, i : i + 1])
                absmag = np.dot(abs(s.T), abs(x[:, i : i + 1]))
                if mag >= 0:
                    signmag = 1.0
                else:
                    signmag = -1.0

                deltas[i] = (
                    signmag
                    * sqrt(eps)
                    * (max(abs(mag), absmag))
                    / ((np.linalg.norm(x[:, i : i + 1]) + np.finfo(float).eps) ** 2)
                )

                if deltas[i] == 0:  # s = 0 or x = 0
                    self.loginfo(
                        "%d-th delta: signmag %g, eps %g, max abs %g, norm %g"
                        % (
                            i,
                            signmag,
                            eps,
                            (max(abs(mag), absmag)),
                            (np.linalg.norm(x) ** 2),
                        )
                    )

                    deltas[i] = sqrt(eps)

                    self.loginfo("%d-th delta: assigned as sqrt(eps) - %g", deltas[i])
                    # raise ValueError('delta is zero? - plz check your
                    # s_init is within a reasonable range')

                # reuse storage x by updating x
                x[:, i : i + 1] = s + deltas[i] * x[:, i : i + 1]

        else:
            for i in range(nruns):
                deltas[i] = delta
                # reuse storage x by updating x
                x[:, i : i + 1] = s + deltas[i] * x[:, i : i + 1]

        simul_obs_purturbation = self.forward_model(x)

        if np.size(simul_obs_purturbation, 1) != nruns:
            raise ValueError(
                "size of simul_obs_purturbation (%d,%d) is not nruns %d"
                % (
                    simul_obs_purturbation.shape[0],
                    simul_obs_purturbation.shape[1],
                    nruns,
                )
            )

        Jxs = np.zeros_like(simul_obs_purturbation)

        # solve Hx HZ HQT
        for i in range(nruns):
            Jxs[:, i : i + 1] = np.true_divide(
                (simul_obs_purturbation[:, i : i + 1] - simul_obs), deltas[i]
            )

        return Jxs

    def objective_function_ls(self, simul_obs) -> float:
        """0.5(y-h(s))^TR^{-1}(y-h(s))"""
        ymhs = (self.obs - simul_obs).ravel()
        return 0.5 * ymhs.T.dot(self.cov_obs_solve(ymhs)).item()

    def objective_function_reg(
        self, s_cur: NDArrayFloat, beta_cur: NDArrayFloat
    ) -> float:
        """0.5(s-Xb)^TC^{-1}(s-Xb)"""
        smxb = (s_cur - np.dot(self.drift.mat, beta_cur)).ravel()
        return float(0.5 * smxb.T.dot(self.Q.solve(smxb)).item())

    def objective_function(self, s_cur, beta_cur, simul_obs) -> float:
        """
        0.5(y-h(s))^TR^{-1}(y-h(s)) + 0.5*(s-Xb)^TQ^{-1}(s-Xb)
        """
        return self.objective_function_ls(simul_obs) + self.objective_function_reg(
            s_cur, beta_cur
        )

    def objective_function_no_beta_new(self, s_cur, simul_obs) -> float:
        """
        marginalized objective w.r.t. beta
        0.5(y-h(s))^TR^{-1}(y-h(s)) + 0.5*(s-Xb)^TQ^{-1}(s-Xb)

        Note: this is an alternative way, more expensive.
        """
        X = self.drift.mat

        def fun(beta: NDArrayFloat) -> float:
            """-X^TC^{-1}(s-Xb)"""
            smxb = (s_cur - np.dot(X, np.atleast_2d(beta))).ravel()
            return float(0.5 * smxb.T.dot(self.Q.solve(smxb)).item())

        # We solve with a newton to find the optimal alpha
        def jac_wrt_beta(beta: NDArrayFloat) -> NDArrayFloat:
            """-X^TC^{-1}(s-Xb)"""
            smxb = (s_cur - np.dot(X, np.atleast_2d(beta))).ravel()
            return -X.T.dot(self.Q.solve(smxb))

        hess = X.T.dot(self.Q.solve(X))

        def hess_wrt_beta(beta: NDArrayFloat) -> NDArrayFloat:
            """X^TC^{-1}X"""
            return hess

        res = sp.optimize.minimize(
            x0=sp.linalg.lstsq(X, s_cur)[0].ravel(),
            fun=fun,
            method="trust-exact",
            jac=jac_wrt_beta,
            hess=hess_wrt_beta,
        )
        self.loginfo(f"new reg part = {res.fun}")
        return self.objective_function_ls(simul_obs) + res.fun

    def objective_function_no_beta(self, s_cur, simul_obs) -> float:
        """
        marginalized objective w.r.t. beta
        0.5(y-h(s))^TR^{-1}(y-h(s)) + 0.5*(s-Xb)^TQ^{-1}(s-Xb)
        """
        X = self.drift.mat
        self.drift.beta_dim

        invZs = np.multiply(
            1.0 / np.sqrt(self.Q.eig_vals), np.dot(self.Q.eig_vects.T, s_cur)
        )
        invZX = np.multiply(
            1.0 / np.sqrt(self.Q.eig_vals), np.dot(self.Q.eig_vects.T, X)
        )
        XTinvQs = np.dot(invZX.T, invZs)
        XTinvQX = np.dot(invZX.T, invZX)
        tmp = np.linalg.solve(
            XTinvQX, XTinvQs
        )  # inexpensive solve p by p where p <= 3, usually p = 1 (scalar division)
        return float(
            (
                self.objective_function_ls(simul_obs)
                + 0.5 * (np.dot(invZs.T, invZs) - np.dot(XTinvQs.T, tmp))
            ).item()
        )

    def rmse(self, residuals: NDArrayFloat, is_normalized: bool) -> float:
        """Return the root mean square error."""
        if is_normalized:
            return np.sqrt(residuals.dot(self.cov_obs_solve(residuals)) / self.d_dim)
        return np.linalg.norm(residuals) / np.sqrt(self.d_dim)

    def jac_mat(self, s_cur, simul_obs, Z):
        m: int = self.s_dim
        p: int = self.drift.beta_dim
        n_pc: int = self.Q.n_pc
        eps: float = self.eps

        temp = np.zeros((m, p + n_pc + 1), dtype="d")  # [HX, HZ, Hs]

        temp[:, 0:p] = np.copy(self.drift.mat)
        temp[:, p : p + n_pc] = np.copy(Z)
        temp[:, p + n_pc : p + n_pc + 1] = np.copy(s_cur)

        Htemp = self.jac_vect(temp, s_cur, simul_obs, eps)

        HX = Htemp[:, 0:p]
        HZ = Htemp[:, p : p + n_pc]
        Hs = Htemp[:, p + n_pc : p + n_pc + 1]

        if self.is_save_jac:
            self.HX = HX
            self.HZ = HZ
            self.Hs = Hs

        # compute the pre-posterior data space
        if p == 1:
            U_data = HX / np.linalg.norm(HX)
        elif p > 1:
            from scipy.linalg import svd

            U_data = svd(
                HX, full_matrices=False, compute_uv=True, lapack_driver="gesdd"
            )[0]
        else:  # point prior
            raise NotImplementedError
        return HX, HZ, Hs, U_data

    def direct_solve(
        self,
        s_cur: NDArrayFloat,
        simul_obs: NDArrayFloat,
        is_use_cholesky: bool = False,
    ):
        """
        Solve the geostatistical system using a direct solver.
        Not to be used unless the number of measurements are small O(100)
        """
        self.loginfo("use direct solver for saddle-point (cokrigging) system")
        m = self.s_dim
        n = self.d_dim
        p = self.drift.beta_dim
        n_pc = self.Q.n_pc
        R = self.cov_obs

        Z = np.sqrt(self.Q.eig_vals).T * self.Q.eig_vects
        # Compute Jacobian-Matrix products
        start1: float = time()
        HX, HZ, Hs, U_data = self.jac_mat(s_cur, simul_obs, Z)

        # Compute eig(P*HQHT*P) approximately by svd(P*HZ)
        start2 = time()

        def mv(v):
            # P*HZ*x = ((I-(U_data*U_data.T))*HZ)*x '''
            tmp = np.dot(HZ, v)
            y = tmp - np.dot(U_data, np.dot(U_data.T, tmp))
            return y

        def rmv(v):
            return np.dot(HZ.T, v) - np.dot(HZ.T, np.dot(U_data, np.dot(U_data.T, v)))

        # Matrix handle for sqrt of Generalized Data Covariance
        sqrtGDCovfun = LinearOperator(
            shape=(n, n_pc), matvec=mv, rmatvec=rmv, dtype="d"
        )
        if self.Q.n_pc <= n - p:
            k = self.Q.n_pc - 1
            _maxiter = n - p
        else:
            k = n - p
            _maxiter = self.Q.n_pc
        sigma_cR = svds(
            sqrtGDCovfun,
            k=k,
            which="LM",
            maxiter=_maxiter,
            return_singular_vectors=False,
            random_state=self.random_state,
        )

        self.loginfo(
            f"computed Jacobian-Matrix products in : {(start2- start1):.3e} secs"
        )

        # Construct HQ directly
        HQ = np.dot(HZ, Z.T)

        if self.is_lm:
            self.loginfo(
                "Solve geostatistical inversion problem (co-kriging, "
                "saddle point systems) with Levenberg-Marquardt"
            )
            nopts = self.nopts_lm
            alpha = 10 ** (np.linspace(0.0, np.log10(self.alphamax_lm), nopts))

        else:
            self.loginfo(
                "Solve geostatistical inversion problem (co-kriging, "
                "saddle point systems)"
            )
            nopts = 1
            alpha = np.array([1.0])

        beta_all = np.zeros((p, nopts), "d")
        s_hat_all = np.zeros((m, nopts), "d")
        Q2_all = np.zeros((1, nopts), "d")
        cR_all = np.zeros((1, nopts), "d")
        LM_eval = np.zeros(nopts, dtype=bool)

        # if LM_smax, LM_smin defined and solution violates them, LM_eval[i] "
        # "becomes True

        for i in range(nopts):  # sequential evaluation for now
            # Construct Psi directly
            # 1) HQH^{t}
            Psi: NDArrayFloat = np.dot(HZ, HZ.T)
            # 2) HQH^{t} + R
            Ri = alpha[i] * R
            if Ri.ndim == 1:
                # Ri is diagonal
                np.fill_diagonal(Psi, Psi.diagonal() + Ri)
            else:
                # If Ri is 2D
                Psi += Ri

            b = np.zeros((n + p, 1), dtype="d")
            # Ax = b, b = obs - h(s) + Hs
            b[:n] = self.obs[:] - simul_obs + Hs[:]

            if is_use_cholesky:
                LA = self.build_cholesky(Psi, HX)
                x = self.solve_cholesky(LA, b, self.Q.n_pc)
            else:
                A = self.build_dense_A(Psi, HX)
                # Create matrix system and solve it
                # cokriging matrix
                x = sp.linalg.solve(A, b)

            # Extract components and return final solution
            # x dimension (n+p,1)
            xi = x[0:n, :]
            beta_all[:, i : i + 1] = x[n : n + p, :]
            s_hat_all[:, i : i + 1] = np.dot(
                self.drift.mat, beta_all[:, i : i + 1]
            ) + np.dot(HQ.T, xi)

            # check prescribed solution range for LM evaluations
            if self.lm_smin is not None:
                if s_hat_all[:, i : i + 1].min() <= self.lm_smin:
                    LM_eval[i] = True
            if self.lm_smax is not None:
                if s_hat_all[:, i : i + 1].max() >= self.lm_smax:
                    LM_eval[i] = True

            # TODO: fix this (21/09/2024)
            Q2_all[:, i : i + 1] = np.dot(b[:n].T, xi) / (n - p)
            tmp_cR = self.get_cR(i, alpha, sigma_cR)
            cR_all[:, i : i + 1] = Q2_all[:, i : i + 1] * np.exp(
                np.log(tmp_cR).sum() / (n - p)
            )

        # evaluate solutions
        if self.is_lm:
            self.loginfo("evaluate LM solutions")
            simul_obs_all = -10000.0 * np.ones((n, nopts), "d")
            s_hat_all_tmp = s_hat_all[:, np.invert(LM_eval)]
            simul_obs_all_tmp = self.forward_model(s_hat_all_tmp)
            simul_obs_all[:, np.invert(LM_eval)] = simul_obs_all_tmp
        else:
            self.loginfo("evaluate the best solution")
            simul_obs_all = self.forward_model(s_hat_all)

        if np.size(simul_obs_all, 1) != nopts:
            raise ValueError("np.size(simul_obs_all,1) != nopts")

        obj_best = 1.0e20
        self.loginfo("%d objective value evaluations" % nopts)
        for i in range(nopts):
            if LM_eval[i]:
                obj = 1.0e20
            else:
                obj = self.objective_function(
                    s_hat_all[:, i : i + 1],
                    beta_all[:, i : i + 1],
                    simul_obs_all[:, i : i + 1],
                )

            if obj < obj_best:
                self.loginfo("%d-th solution obj %e (alpha %f)" % (i, obj, alpha[i]))
                s_hat = s_hat_all[:, i : i + 1]
                beta = beta_all[:, i : i + 1]
                simul_obs_new = simul_obs_all[:, i : i + 1]
                obj_best = obj

        return s_hat, beta, simul_obs_new

    # TODO: fix this (21/09/2024)
    def get_cR(
        self, i: int, alpha: NDArrayFloat, sigma_cR: NDArrayFloat
    ) -> NDArrayFloat:
        n = self.d_dim
        p = self.drift.beta_dim
        R = self.cov_obs

        tmp_cR = np.zeros((n - p, 1), np.float64)

        if R.size == 1:  # single observation variance
            # self.loginfo(f"alpha[{i}] = {alpha[i]}")
            tmp_cR[:] = alpha[i] * R  # scalar placed in all meshes
            tmp_cR[: sigma_cR.shape[0]] = (
                tmp_cR[: sigma_cR.shape[0]] + (sigma_cR[:, np.newaxis]) ** 2
            )
        elif R.ndim == 1:  # diagonal covariance matrix
            tmp_cR = np.multiply(alpha[i], R[:-p])

            # self.loginfo(f"tmp_cR.shape = {tmp_cR.shape}")
            # self.loginfo(f"tmp_cR.shape = {tmp_cR.shape}")

            uniqueR = np.unique(R)
            lenR = len(uniqueR)
            lenRi = int((n - sigma_cR.shape[0]) / lenR)
            strtidx = sigma_cR.shape[0]
            strtidx = +0  # to remove, Antoine 22/09/2024
            # self.loginfo(f"lenRi = {lenRi}")

            # # this loop works only if self.is_lm == True, otherwise, alpha is a
            # scalar, there is an issue somewhere.
            for iR in range(lenR):
                tmp_cR[strtidx : strtidx + lenRi] = (
                    alpha[min(iR, alpha.size - 1)] * uniqueR[iR]
                )
                strtidx = strtidx + lenRi
            tmp_cR[strtidx:] = alpha[min(iR, alpha.size - 1)] * uniqueR[iR]
        else:  # symmetrical square covariance matrix
            pass

        tmp_cR[tmp_cR <= 0] = 1.0e-16  # temporary fix for zero tmp_cR

        return tmp_cR

    def iterative_solve(self, s_cur: NDArrayFloat, simul_obs: NDArrayFloat):
        """
        Iterative Solve
        """
        m = self.s_dim
        n = self.d_dim
        p = self.drift.beta_dim
        n_pc = self.Q.n_pc
        R = self.cov_obs

        Z = np.sqrt(self.Q.eig_vals).T * self.Q.eig_vects

        # Compute Jacobian-Matrix products
        start1 = time()
        HX, HZ, Hs, U_data = self.jac_mat(s_cur, simul_obs, Z)
        # debug_here()

        start2 = time()
        self.loginfo("computed Jacobian-Matrix products in %f secs" % (start2 - start1))

        # Compute Q2/cR for covariance model validation
        if (
            self.cov_obs.shape[0] == 1
        ):  # Compute eig(P*(HQHT+R)*P) approximately by svd(P*HZ)**2 + R if
            # R is single number

            def mv(v):
                # P*HZ*x = ((I-(U_data*U_data.T))*HZ)*x '''
                tmp = np.dot(HZ, v)
                y = tmp - np.dot(U_data, np.dot(U_data.T, tmp))
                return y

            def rmv(v):
                return np.dot(HZ.T, v) - np.dot(
                    HZ.T, np.dot(U_data, np.dot(U_data.T, v))
                )

            # Matrix handle for sqrt of Generalized Data Covariance
            sqrtGDCovfun = LinearOperator(
                shape=(n, n_pc), matvec=mv, rmatvec=rmv, dtype="d"
            )

            # sigma_cR = svds(sqrtGDCovfun, k= min(n-p-1,n_pc-1), which='LM',
            # maxiter = n, return_singular_vectors=False)

            if n_pc <= n - p:
                k = n_pc - 1
                max_iter = n
            else:
                k = n - p
                max_iter = n_pc

            sigma_cR = svds(
                sqrtGDCovfun,
                k=k,
                which="LM",
                maxiter=max_iter,
                return_singular_vectors=False,
                random_state=self.random_state,
            )

            self.loginfo(
                "eig. val. of generalized data covariance : "
                "%f secs (%8.2e, %8.2e, %8.2e)"
                % (time() - start2, sigma_cR[0], sigma_cR.min(), sigma_cR.max())
            )
            # of generalized data covariance : %f secs (%8.2e, %8.2e, %8.2e)"
            # % (start2 - start1, time()-start2,sigma_cR[0],sigma_cR.min(),
            # sigma_cR.max()))
        else:  # Compute eig(P*(HQHT+R)*P) approximately by svd(P*(HZ*HZ' + R)*P)
            # need to do for each alpha[i]*R
            pass

        # preconditioner construction
        # will add more description here
        if self.is_use_preconditioner:
            tStart_precond = time()

            # GHEP : HQHT u = lamdba R u => u = R^{-1/2} y
            # original implementation was sqrt of R^{-1/2} HZ n by n_pc
            # svds cannot compute entire n_pc eigenvalues so do this for
            # n by n matrix
            # this leads to double the cost
            # TODO: this is not working for 2D R matrix
            def pmv(v):
                return np.multiply(
                    1 / np.sqrt(R),
                    np.dot(HZ, (np.dot(HZ.T, np.multiply(1 / np.sqrt(R), v)))),
                )
                # TODO: why is it different ?
                # return np.multiply(self.invsqrtR,np.dot(HZ,v))

            def prmv(v):
                return pmv(v)

            # self.loginfo('preconditioner construction using Generalized
            # Eigen-decomposition')
            #    self.loginfo("n :%d & n_pc: %d" % (n,n_pc))

            ## Matrix handle for sqrt of Data Covariance
            ##sqrtDataCovfun = LinearOperator( (n,n_pc), matvec=pmv,
            # rmatvec = prmv, dtype = 'd')
            ##sqrtDataCovfun = LinearOperator((n, n), matvec=pmv,
            # rmatvec=prmv, dtype='d')
            ##[Psi_U,Psi_sigma,Psi_V] = svds(sqrtDataCovfun,
            # k= min(n,n_pc), which='LM', maxiter = n, return_singular_vectors='u')

            # Matrix handle for Data Covariance
            DataCovfun = LinearOperator((n, n), matvec=pmv, rmatvec=prmv, dtype="d")

            if n_pc < n:
                [Psi_sigma, Psi_U] = eigsh(
                    DataCovfun, k=n_pc, which="LM", maxiter=n, v0=self.get_v0(n)
                )
            elif n_pc == n:
                [Psi_sigma, Psi_U] = eigsh(
                    DataCovfun, k=n_pc - 1, which="LM", maxiter=n, v0=self.get_v0(n)
                )
            else:
                [Psi_sigma, Psi_U] = eigsh(
                    DataCovfun, k=n - 1, which="LM", maxiter=n_pc, v0=self.get_v0(n)
                )

            # self.loginfo("eig. val. of sqrt data covariance (%8.2e, %8.2e, %8.2e)"
            # % (Psi_sigma[0], Psi_sigma.min(), Psi_sigma.max()))
            # self.loginfo(Psi_sigma)

            # TODO: change
            Psi_U = np.multiply(1 / np.sqrt(R).reshape(-1, 1), Psi_U)
            # if R.shape[0] == 1:
            # Psi_sigma = Psi_sigma**2 # because we use svd(HZ)
            # instead of svd(HQHT+R)
            index_Psi_sigma = np.argsort(Psi_sigma)
            index_Psi_sigma = index_Psi_sigma[::-1]
            Psi_sigma = Psi_sigma[index_Psi_sigma]
            Psi_U = Psi_U[:, index_Psi_sigma]
            Psi_U = Psi_U[:, Psi_sigma > 0]
            Psi_sigma = Psi_sigma[Psi_sigma > 0]

            self.loginfo(
                "time for data covarance construction : %f sec "
                % (time() - tStart_precond)
            )
            self.loginfo(
                "eig. val. of data covariance (%8.2e, %8.2e, %8.2e)"
                % (Psi_sigma[0], Psi_sigma.min(), Psi_sigma.max())
            )
            if Psi_U.shape[1] != n_pc:
                self.loginfo(
                    "- rank of data covariance :%d for preconditioner construction"
                    % (Psi_U.shape[1])
                )

            self.Psi_sigma = Psi_sigma
            self.Psi_U = Psi_U

        if self.is_lm:
            self.loginfo(
                "solve saddle point (co-kriging) systems with Levenberg-Marquardt"
            )
            nopts = self.nopts_lm
            alpha = 10 ** (np.linspace(0.0, np.log10(self.alphamax_lm), nopts))
        else:
            self.loginfo("solve saddle point (co-kriging) system")
            nopts = 1
            alpha = np.array([1.0])

        beta_all = np.zeros((p, nopts), "d")
        s_hat_all = np.zeros((m, nopts), "d")
        Q2_all = np.zeros((1, nopts), "d")
        cR_all = np.zeros((1, nopts), "d")
        LM_eval = np.zeros(nopts, dtype=bool)
        # if LM_smax, LM_smin defined and solution violates them, LM_eval[i]
        # becomes True
        for i in range(nopts):  # this is sequential for now
            # Create matrix context for cokriging matrix-vector multiplication
            if R.shape[0] == 1:

                def mv(v: NDArrayFloat) -> NDArrayFloat:
                    return np.concatenate(
                        (
                            (
                                np.dot(HZ, np.dot(HZ.T, v[0:n]))
                                + np.multiply(np.multiply(alpha[i], R), v[0:n])
                                + np.dot(HX, v[n : n + p])
                            ),
                            (np.dot(HX.T, v[0:n])),
                        ),
                        axis=0,
                    )

            else:

                def mv(v: NDArrayFloat) -> NDArrayFloat:
                    return np.concatenate(
                        (
                            (
                                np.dot(HZ, np.dot(HZ.T, v[0:n]))
                                + np.multiply(
                                    np.multiply(alpha[i], R.reshape(v[0:n].shape)),
                                    v[0:n],
                                )
                                + np.dot(HX, v[n : n + p])
                            ),
                            (np.dot(HX.T, v[0:n])),
                        ),
                        axis=0,
                    )

            # Matrix handle
            Afun = LinearOperator(
                shape=(n + p, n + p), matvec=mv, rmatvec=mv, dtype="d"
            )

            b = np.zeros((n + p, 1), dtype="d")
            b[:n] = self.obs[:] - simul_obs + Hs[:]

            callback = Residual()

            # Residual and maximum iterations
            # TODO: parametrize
            itertol = 1.0e-10
            solver_maxiter = self.s_dim
            # itertol = (
            #     1.0e-10
            #     if "iterative_tol" not in self.params
            #     else self.params["iterative_tol"]
            # )
            # solver_maxiter = (
            #     m
            #     if "iterative_maxiter" not in self.params
            #     else self.params["iterative_maxiter"]
            # )

            # construction preconditioner
            if self.is_use_preconditioner:
                #
                # Lee et al. WRR 2016 Eq 16 - 21, Saibaba et al. NLAA 2015
                # R_LM = alpha * R
                # Psi_U_LM = 1./sqrt(alpha) * Psi_U
                # Psi_sigma = Psi_sigma/alpha
                #
                # (R^-1 - UDvecU')*v

                if R.shape[0] == 1:

                    def invPsi(v):
                        Dvec = np.divide(
                            (1.0 / alpha[i] * Psi_sigma),
                            ((1.0 / alpha[i]) * Psi_sigma + 1.0),
                        )  # (n_pc,)
                        Psi_U_i = np.multiply(
                            (1.0 / sqrt(alpha[i])), Psi_U
                        )  # (n, n_pc) (dim[1] can be n_pc-1, n)
                        Psi_UTv = np.dot(
                            Psi_U_i.T, v
                        )  # n_pc by n * v (can be (n,) or (n,p)) = (n_pc,) or (n_pc,p)

                        # TODO: remove these stupid reshapes
                        alphainvRv = (1.0 / alpha[i]) * self.cov_obs_solve(v).reshape(
                            -1, 1
                        )

                        if Psi_UTv.ndim == 1:
                            PsiDPsiTv = np.dot(
                                Psi_U_i,
                                np.multiply(
                                    Dvec[: Psi_U_i.shape[1]].reshape(Psi_UTv.shape),
                                    Psi_UTv,
                                ),
                            )
                        elif Psi_UTv.ndim == 2:  # for invPsi(HX)
                            DMat = np.tile(
                                Dvec[: Psi_U_i.shape[1]], (Psi_UTv.shape[1], 1)
                            ).T  # n_pc by p
                            PsiDPsiTv = np.dot(Psi_U_i, np.multiply(DMat, Psi_UTv))
                        else:
                            raise ValueError(
                                "Psi_U times vector should have a dimension smaller "
                                "than 2 - current dim = %d" % (Psi_UTv.ndim)
                            )

                        return alphainvRv - PsiDPsiTv

                else:

                    def invPsi(v):
                        Dvec = np.divide(
                            (1.0 / alpha[i] * Psi_sigma),
                            ((1.0 / alpha[i]) * Psi_sigma + 1.0),
                        )
                        Psi_U_i = np.multiply((1.0 / sqrt(alpha[i])), Psi_U)
                        Psi_UTv = np.dot(Psi_U_i.T, v)
                        # TODO: remove these stupid reshape by using vectors
                        # and matrices
                        alphainvRv = (1.0 / alpha[i]) * self.cov_obs_solve(v)
                        if Psi_UTv.ndim == 1:
                            PsiDPsiTv = np.dot(
                                Psi_U_i,
                                np.multiply(
                                    Dvec[: Psi_U_i.shape[1]].reshape(Psi_UTv.shape),
                                    Psi_UTv,
                                ),
                            )
                        elif Psi_UTv.ndim == 2:  # for invPsi(HX)
                            alphainvRv = (
                                (1.0 / alpha[i]) * self.cov_obs_solve(v.ravel())
                            ).reshape(-1, 1)
                            Dmat = np.tile(
                                Dvec[: Psi_U_i.shape[1]], (Psi_UTv.shape[1], 1)
                            ).T  # n_pc by p
                            PsiDPsiTv = np.dot(Psi_U_i, np.multiply(Dmat, Psi_UTv))
                        else:
                            raise ValueError(
                                "Psi_U times vector should have a dimension "
                                "smaller than 2 - current dim = %d" % (Psi_UTv.ndim)
                            )

                        return alphainvRv - PsiDPsiTv

                # Preconditioner construction Lee et al. WRR 2016 Eq (14)
                # typo in Eq (14), (2,2) block matrix should be -S^-1 instead of -S
                def Pmv(v):
                    invPsiv = invPsi(v[0:n])
                    S = np.dot(HX.T, invPsi(HX))  # p by p matrix
                    invSHXTinvPsiv = np.linalg.solve(S, np.dot(HX.T, invPsiv))
                    invPsiHXinvSHXTinvPsiv = invPsi(np.dot(HX, invSHXTinvPsiv))
                    invPsiHXinvSv1 = invPsi(np.dot(HX, np.linalg.solve(S, v[n:])))
                    invSv1 = np.linalg.solve(S, v[n:])
                    return np.concatenate(
                        (
                            (invPsiv - invPsiHXinvSHXTinvPsiv + invPsiHXinvSv1),
                            (invSHXTinvPsiv - invSv1),
                        ),
                        axis=0,
                    )

                P = LinearOperator(
                    shape=(n + p, n + p), matvec=Pmv, rmatvec=Pmv, dtype="d"
                )

                # TODO: parametrize
                restart = 50
                x, info = gmres(
                    Afun,
                    b,
                    restart=restart,
                    maxiter=solver_maxiter,
                    callback=callback,
                    M=P,
                    atol=itertol,
                    rtol=itertol,
                    callback_type="legacy",
                )
                self.loginfo(
                    "-- Number of iterations for gmres %g" % (callback.itercount())
                )
                if info != 0:  # if not converged
                    callback = Residual()
                    x, info = minres(
                        Afun,
                        b,
                        x0=x,
                        rtol=itertol,
                        maxiter=solver_maxiter,
                        callback=callback,
                        M=P,
                    )
                    self.loginfo(
                        "-- Number of iterations for minres %g and info %d"
                        % (callback.itercount(), info)
                    )
            else:
                x, info = minres(
                    Afun, b, rtol=itertol, maxiter=solver_maxiter, callback=callback
                )
                self.loginfo(
                    "-- Number of iterations for minres %g" % (callback.itercount())
                )

                if info != 0:
                    x, info = gmres(
                        Afun,
                        b,
                        x0=x,
                        rtol=itertol,
                        maxiter=solver_maxiter,
                        callback=callback,
                        atol=itertol,
                        callback_type="legacy",
                    )
                    self.loginfo(
                        "-- Number of iterations for gmres: %g, info: %d, tol: %f"
                        % (callback.itercount(), info, itertol)
                    )

            # Extract components and postprocess
            # x.shape = (n+p,), so need to increase the dimension (n+p,1)
            xi = x[0:n, np.newaxis]
            beta_all[:, i : i + 1] = x[n : n + p, np.newaxis]

            # from IPython.core.debugger import Tracer; debug_here = Tracer()
            s_hat_all[:, i : i + 1] = np.dot(
                self.drift.mat, beta_all[:, i : i + 1]
            ) + np.dot(Z, np.dot(HZ.T, xi))

            # check prescribed solution range for LM evaluations
            if self.lm_smin is not None:
                if s_hat_all[:, i : i + 1].min() <= self.lm_smin:
                    LM_eval[i] = True
            if self.lm_smax is not None:
                if s_hat_all[:, i : i + 1].max() >= self.lm_smax:
                    LM_eval[i] = True

            if LM_eval[i]:
                self.loginfo(
                    "%d - min(s): %g, max(s) :%g - violate LM_smin or LM_smax"
                    % (
                        i,
                        s_hat_all[:, i : i + 1].min(),
                        s_hat_all[:, i : i + 1].max(),
                    )
                )
            else:
                self.loginfo(
                    "%d - min(s): %g, max(s) :%g"
                    % (
                        i,
                        s_hat_all[:, i : i + 1].min(),
                        s_hat_all[:, i : i + 1].max(),
                    )
                )

            Q2_all[:, i : i + 1] = np.dot(b[:n].T, xi) / (n - p)

            # model validation, predictive diagnostics cR/Q2
            if R.shape[0] == 1:
                tmp_cR = np.zeros((n - p, 1), "d")
                tmp_cR[:] = np.multiply(alpha[i], R)
                tmp_cR[: sigma_cR.shape[0]] = (
                    tmp_cR[: sigma_cR.shape[0]] + (sigma_cR[:, np.newaxis]) ** 2
                )
            else:
                # need to find efficient way to compute cR once
                # approximation
                def mv(v):
                    # P*(HZ*HZ.T + R)*P*x = P = (I-(U_data*U_data.T))
                    # debug_here()
                    Pv = v - np.dot(U_data, np.dot(U_data.T, v))  # P * v : n by 1
                    RPv = np.multiply(
                        alpha[i], np.multiply(R.reshape(v.shape), Pv)
                    )  # alpha*R*P*v : n by 1
                    PRPv = RPv - np.dot(
                        U_data, np.dot(U_data.T, RPv)
                    )  # P*R*P*v : n by 1
                    HQHTPv = np.dot(HZ, np.dot(HZ.T, Pv))  # HQHTPv : n by 1
                    PHQHTPv = HQHTPv - np.dot(
                        U_data, np.dot(U_data.T, HQHTPv)
                    )  # P*HQHT*P*v
                    return PHQHTPv + PRPv

                def rmv(v):
                    return mv(v)  # symmetric matrix

                # Matrix handle for Generalized Data Covariance
                sqrtGDCovfun = LinearOperator((n, n), matvec=mv, rmatvec=rmv, dtype="d")

                if n_pc < n - p:
                    k: int = n_pc
                    maxiter: int = n - p
                elif n_pc == n - p:
                    k = n_pc - 1
                    maxiter = n - p
                else:
                    k = n - p
                    maxiter = n_pc

                sigma_cR = svds(
                    sqrtGDCovfun,
                    k=k,
                    which="LM",
                    maxiter=maxiter,
                    return_singular_vectors=False,
                    random_state=self.random_state,
                )

                tmp_cR = np.zeros((n - p, 1), "d")
                tmp_cR[:] = np.multiply(alpha[i], R[:-p]).reshape(
                    -1, 1
                )  # TODO: remove the reshape

                tmp_cR[: sigma_cR.shape[0]] = sigma_cR[:, np.newaxis]

                uniqueR = np.unique(R)
                lenR = len(uniqueR)
                lenRi = int((n - sigma_cR.shape[0]) / lenR)
                strtidx = sigma_cR.shape[0]
                for iR in range(lenR):
                    tmp_cR[strtidx : strtidx + lenRi] = alpha[iR] * uniqueR[iR]
                    strtidx = strtidx + lenRi
                tmp_cR[strtidx:] = alpha[iR] * uniqueR[iR]

            tmp_cR[tmp_cR <= 0] = 1.0e-16  # temporary fix for zero tmp_cR
            cR_all[:, i : i + 1] = Q2_all[:, i : i + 1] * np.exp(
                np.log(tmp_cR).sum() / (n - p)
            )

        # evaluate solutions
        if self.is_lm:
            self.loginfo("evaluate LM solutions")
            simul_obs_all = -10000.0 * np.ones((n, nopts), "d")
            s_hat_all_tmp = s_hat_all[:, np.invert(LM_eval)]
            simul_obs_all_tmp = self.forward_model(s_hat_all_tmp)
            self.loginfo("LM solution evaluated")
            simul_obs_all[:, np.invert(LM_eval)] = simul_obs_all_tmp

        else:
            self.loginfo("evaluate the best solution")
            simul_obs_all = self.forward_model(s_hat_all)

        if np.size(simul_obs_all, 1) != nopts:
            return ValueError("np.size(simul_obs_all,1) should be nopts")

        # evaluate objective values and select best value
        obj_best = 1.0e20
        if self.is_lm:
            self.loginfo("%d objective value evaluations" % nopts)

        i_best = -1

        for i in range(nopts):
            if LM_eval[i]:
                obj = 1.0e20
            else:
                obj = self.objective_function(
                    s_hat_all[:, i : i + 1],
                    beta_all[:, i : i + 1],
                    simul_obs_all[:, i : i + 1],
                )

            if obj < obj_best:
                s_hat = s_hat_all[:, i : i + 1]
                beta = beta_all[:, i : i + 1]
                simul_obs_new = simul_obs_all[:, i : i + 1]
                obj_best = obj
                i_best = i
                self.loginfo(
                    f"{i:d}-th solution obj {obj} (alpha {alpha[i]}, beta {beta})"
                )

        if i_best == -1:
            self.loginfo("no better solution found ..")
            s_hat = s_cur
            simul_obs_new = simul_obs
            beta = 0.0

        if self.post_cov_estimation is not None:
            self.HZ = HZ
            self.HX = HX
            self.R_LM = self.cov_obs * alpha[i_best]

        self.istate.i_best = i_best  # keep track of best LM solution
        return s_hat, beta, simul_obs_new

    def linear_iteration(
        self, s_cur: NDArrayFloat, simul_obs: NDArrayFloat, n_iter: int
    ):
        # Solve geostatistical system -> two ways, direct of iterative solve
        if self.is_direct_solve:
            s_hat, beta, simul_obs_new = self.direct_solve(s_cur, simul_obs)
        else:
            s_hat, beta, simul_obs_new = self.iterative_solve(s_cur, simul_obs)
        obj: float = self.objective_function(s_hat, beta, simul_obs_new)

        # Call the optional callback at the end of each linear iteration so some
        # intermediate solver states could be saved
        if self.callback is not None:
            self.callback(self, s_hat=s_hat, simul_obs=simul_obs, n_iter=n_iter)

        return s_hat, beta, simul_obs_new, obj

    def line_search(self, s_cur, s_past):
        nopts = self.nopts_lm
        m = self.s_dim

        s_hat_all = np.zeros((m, nopts), "d")
        # need to remove delta = 0 and 1
        delta = np.linspace(-0.1, 1.1, nopts)

        for i in range(nopts):
            s_hat_all[:, i : i + 1] = delta[i] * s_past + (1.0 - delta[i]) * s_cur

        self.loginfo("evaluate linesearch solutions")
        simul_obs_all = self.forward_model(s_hat_all)

        # will change assert to valueerror
        assert np.size(simul_obs_all, 1) == nopts
        obj_best = 1.0e20

        for i in range(nopts):
            obj = self.objective_function_no_beta(
                s_hat_all[:, i : i + 1], simul_obs_all[:, i : i + 1]
            )

            if obj < obj_best:
                self.loginfo("%d-th solution obj %e (delta %f)" % (i, obj, delta[i]))
                s_hat = s_hat_all[:, i : i + 1]
                simul_obs_new = simul_obs_all[:, i : i + 1]
                obj_best = obj

        return s_hat, simul_obs_new, obj_best

    def display_objfun(
        self,
        loss_ls: float,
        n_obs: int,
        rmse: float,
        n_rmse: float,
        n_iter: int = 0,
        obj: Optional[float] = None,
        res: Optional[float] = None,
        is_beta: bool = True,
    ) -> None:
        if n_iter != 0:
            self.loginfo(f"== iteration {n_iter + 1:d} summary ==")

        dat = {
            "LS objfun 0.5 (obs. diff.)^T R^{-1}(obs. diff.)": loss_ls,
            "norm LS objfun 0.5 / nobs (obs. diff.)^T R^{-1}(obs. diff.)": loss_ls
            / n_obs,
            "RMSE (norm(obs. diff.)/sqrt(nobs))": rmse,
            "norm RMSE (norm(obs. diff./sqrtR)/sqrt(nobs))": n_rmse,
        }
        if obj is not None:
            if is_beta:
                dat["objective function"] = obj
            else:
                dat["objective function (no beta)"] = obj
        if res is not None:
            dat[f"relative L2-norm diff btw sol {n_iter:d} and sol {n_iter+1:d}"] = res

        maxlen = max([len(k) for k in dat]) + 1
        for k, v in dat.items():
            self.loginfo(f"** {k:<{maxlen}} : {v:.3e}")

    def gauss_newton(self) -> Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, int]:
        """
        Gauss-newton iteration
        """

        s_init = self.s_init
        self.maxiter

        res = 1.0

        self.loginfo("##### 2. Start PCGA Inversion #####")
        self.loginfo("-- evaluate initial solution")

        # s_init has shape (s_dim, 1)
        simul_obs_init = self.forward_model(s_init)
        self.istate.simul_obs_best = simul_obs_init

        self.simul_obs_init = simul_obs_init
        residuals = (simul_obs_init - self.obs).ravel()
        rmse_init: float = self.rmse(residuals, False)
        n_rmse_init: float = self.rmse(residuals, True)
        loss_ls_init = self.objective_function_ls(simul_obs_init)

        simul_obs_cur = np.copy(simul_obs_init)
        s_cur = np.copy(s_init)
        s_past = np.copy(s_init)

        # initial objective function -> very high
        obj = self.objective_function_no_beta(s_cur, simul_obs_cur)

        self.display_objfun(
            loss_ls_init,
            simul_obs_init.size,
            rmse_init,
            n_rmse_init,
            obj=obj,
            is_beta=False,
        )

        # save the initial objective function
        self.istate.objvals.append(float(obj))

        # Save the initial state
        if self.callback is not None:
            self.callback(self, s_hat=s_cur, simul_obs=simul_obs_cur, n_iter=0)

        for n_iter in range(self.maxiter):  # type: ignore
            start = time()

            # TODO: make a loop for that

            self.loginfo(f"***** Iteration {n_iter + 1} ******")
            s_cur, beta_cur, simul_obs_cur, obj = self.linear_iteration(
                s_past, simul_obs_cur, n_iter
            )

            self.loginfo(
                "- Geostat. inversion at iteration %d is %g sec"
                % ((n_iter + 1), round(time() - start))
            )

            # case 1: progress in objective function
            if obj < self.istate.obj_best:
                self.istate.s_best = s_cur
                self.istate.beta_best = beta_cur
                self.istate.simul_obs_best = simul_obs_cur
                self.istate.iter_best = n_iter + 1
            # case 2: no progress in objective function
            else:
                if self.is_line_search:
                    self.loginfo(
                        "perform simple linesearch due to no progress in obj value"
                    )
                    s_cur, simul_obs_cur, obj = self.line_search(s_cur, s_past)
                    if obj < self.istate.obj_best:
                        self.istate.s_best = s_cur
                        self.istate.simul_obs_best = simul_obs_cur
                        self.istate.iter_best = n_iter + 1
                    else:
                        if n_iter > 1:
                            self.loginfo("no progress in obj value")
                            n_iter += 1
                            break
                        else:
                            self.loginfo(
                                "no progress in obj value but wait for one "
                                "more iteration.."
                            )
                            # allow first few iterations
                            pass  # allow for
                else:
                    self.istate.status = "CONVERGENCE: NO PROGRESS IN OBJ VALUE"
                    self.istate.is_success = False
                    n_iter += 1
                    break

            res = float(np.linalg.norm(s_past - s_cur) / np.linalg.norm(s_past))
            residuals = (simul_obs_cur - self.obs).ravel()

            loss_ls = self.objective_function_ls(simul_obs_cur)
            rmse = self.rmse(residuals, False)
            n_rmse = self.rmse(residuals, True)
            obj = self.objective_function(s_cur, beta_cur, simul_obs_cur)

            self.display_objfun(
                loss_ls, simul_obs_init.size, rmse, n_rmse, n_iter, obj=obj, res=res
            )

            if res < self.restol:
                self.istate.status = "CONVERGENCE: MODEL_CHANGE_<=_RES_TOL"
                self.istate.is_success = True
                n_iter += 1
                break
            elif (
                np.abs((obj - self.istate.objvals[-1]))
                / max(abs(self.istate.objvals[-1]), abs(obj), 1)
                < self.ftol
            ):
                self.istate.status = "CONVERGENCE: REL_REDUCTION_OF_F_<=_FTOL"
                self.istate.is_success = True
                n_iter += 1
                break
            elif self.ftarget is not None:
                if self.ftarget > obj:
                    self.istate.status = "CONVERGENCE: F_<=_FACTR*EPSMCH"
                    self.istate.is_success = True
                    n_iter += 1
                    break

            # To add before the previous check, otherwise
            # obj == self.istate.objvals[-1] and the elif condition is always True
            # which cause an early break
            self.istate.objvals.append(float(obj))
            s_past = np.copy(s_cur)

        if n_iter + 1 > self.maxiter and not self.istate.is_success:
            self.istate.status = "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT"

        if self.post_cov_estimation is not None:
            # assume linesearch result close to the current solution
            start = time()
            if self.istate.i_best is None:
                self.istate.i_best = 0
            if self.post_cov_estimation == PostCovEstimation.DIRECT:
                self.loginfo(
                    "start direct posterior variance computation "
                    "- this option works for O(nobs) ~ 100"
                )
                self.post_diagv = self.ComputePosteriorDiagonalEntriesDirect(
                    self.HZ, self.HX, self.istate.i_best, self.cov_obs
                )
            else:
                self.loginfo("start posterior variance computation")
                self.post_diagv = self.ComputePosteriorDiagonalEntries(
                    self.HZ, self.HX, self.istate.i_best, self.cov_obs
                )
            self.loginfo("posterior diag. computed in %f secs" % (time() - start))
            # if self.iter_save:
            #     np.savetxt("./postv.txt", self.post_diagv)

        # return s_cur, beta_cur, simul_obs, iter_cur
        self.loginfo("------------ Inversion Summary ---------------------------")
        self.loginfo(f"** Success = {self.istate.is_success}")
        self.loginfo(f"** Status  = {self.istate.status}")

        if self.istate.iter_best == 0:
            self.loginfo("** Did not found better solution than initial guess")
        else:
            self.loginfo(f"** Found solution at iteration {self.istate.iter_best}")
        residuals = (self.istate.simul_obs_best - self.obs).ravel()
        loss_ls_best = self.objective_function_ls(self.istate.simul_obs_best)
        rmse_best: float = self.rmse(residuals, False)
        n_rmse_best: float = self.rmse(residuals, True)

        self.display_objfun(
            loss_ls_best,
            simul_obs_init.size,
            rmse_best,
            n_rmse_best,
            obj=self.istate.obj_best,
        )

        self.loginfo(
            f"- Final predictive model checking Q2 = "
            f"{[f'{float(Q2[0]):.3e}' for Q2 in self.istate.Q2_best]}"
        )

        self.loginfo(
            f"- Final cR = {[f'{float(cR[0]):.3e}' for cR in self.istate.cR_best]}"
        )

        return (
            self.istate.s_best,
            self.istate.simul_obs_best,
            self.post_diagv,
            self.istate.iter_best,
        )

    def run(self) -> Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, int]:
        start = time()
        s_hat, simul_obs, post_diagv, iter_best = self.gauss_newton()
        self.loginfo(f"** Total elapsed time is {(time() - start):.3e} secs")
        self.loginfo("----------------------------------------------------------")
        return s_hat, simul_obs, post_diagv, iter_best

    def get_psi(
        self, HZ: NDArrayFloat, i_best: int, cov_obs: NDArrayFloat
    ) -> NDArrayFloat:
        """Get the matrix HQH^{T} + R."""

        if self.is_lm:
            # self.loginfo(
            #     "Solve geostatistical inversion problem (co-kriging, "
            #     "saddle point systems) with Levenberg-Marquardt"
            # )
            nopts = self.nopts_lm
            alpha = 10 ** (np.linspace(0.0, np.log10(self.alphamax_lm), nopts))

        else:
            # self.loginfo(
            #     "Solve geostatistical inversion problem (co-kriging, "
            #     "saddle point systems)"
            # )
            nopts = 1
            alpha = np.array([1.0])

        # alpha = 10 ** (np.linspace(0.0, np.log10(self.alphamax_lm), self.nopts_lm))
        Ri = np.multiply(alpha[i_best], cov_obs)

        # Construct Psi directly
        Psi: NDArrayFloat = np.dot(HZ, HZ.T)  # HQH^{t}
        # Add the R matrix
        if Ri.ndim == 1:
            # Ri is diagonal
            np.fill_diagonal(Psi, Psi.diagonal() + Ri)
        else:
            # If Ri is 2D
            Psi += Ri
        return Psi

    @staticmethod
    def build_cholesky(Psi: NDArrayFloat, HX: NDArrayFloat) -> NDArrayFloat:
        # HQH^{T} and R are positive semi-definite
        # Then we just need to factorize L11
        # Cholesky:
        L11 = sp.linalg.cholesky(Psi, lower=True)
        L12 = sp.linalg.solve_triangular(L11, HX, lower=True, trans="N")
        L22 = sp.linalg.cholesky(L12.T @ L12, lower=True)
        return np.hstack(
            [np.vstack([L11, L12.T]), np.vstack([np.zeros(L12.shape), L22])]
        )

    @staticmethod
    def build_dense_A_from_cholesky(LA: NDArrayFloat, n_pc: int) -> NDArrayFloat:
        # We don't build explicitly E, we just build the diagonal values
        Ev = np.ones(LA.shape[0])
        Ev[n_pc + 1 :] *= -1
        return (LA * Ev) @ LA.T

    @staticmethod
    def solve_cholesky(LA: NDArrayFloat, v: NDArrayFloat, n_pc: int) -> NDArrayFloat:
        # LA is the lowest triangle of the cholesky factorization LA @ E @ LA.T.
        Ev = np.ones(LA.shape[0])
        Ev[n_pc + 1 :] *= -1
        v = sp.linalg.solve_triangular(LA * Ev, v, lower=True)
        return sp.linalg.solve_triangular(LA.T, v, lower=False)

    @staticmethod
    def build_dense_A(Psi: NDArrayFloat, HX: NDArrayFloat) -> NDArrayFloat:
        n: int = Psi.shape[0]
        p: int = HX.shape[1]
        A = np.zeros((n + p, n + p), dtype="d")
        A[0:n, 0:n] = np.copy(Psi)
        A[0:n, n : n + p] = np.copy(HX)
        A[n : n + p, 0:n] = np.copy(HX.T)
        return A

    def ComputePosteriorDiagonalEntriesDirect(
        self, HZ, HX, i_best, cov_obs, is_use_cholesky: bool = True
    ) -> NDArrayFloat:
        """Computing posterior diagonal entries using cholesky."""

        Psi = self.get_psi(HZ, i_best, cov_obs)
        Z = np.sqrt(self.Q.eig_vals).T * self.Q.eig_vects

        # [HQ, X^{T}]
        b_all = np.vstack([np.dot(HZ, Z.T), self.drift.mat.T])
        # Use cholesky factorization to solve the system
        LA = self.build_cholesky(Psi, HX)
        return (
            self.prior_s_var
            - np.sum(b_all * self.solve_cholesky(LA, b_all, self.Q.n_pc), axis=0)
        ).reshape(-1, 1)

    def ComputePosteriorDiagonalEntries(self, HZ, HX, i_best, R):
        """
        Computing posterior diagonal entries using iterative approach
        """
        m = self.s_dim
        n = self.d_dim
        p = self.drift.beta_dim

        alpha = 10 ** (np.linspace(0.0, np.log10(self.alphamax_lm), self.nopts_lm))

        ## Create matrix context
        # if R.shape[0] == 1:
        #    def mv(v):
        #        return np.concatenate(((np.dot(HZ, np.dot(HZ.T, v[0:n])) +
        # np.multiply(np.multiply(alpha[i_best], R),v[0:n])
        # + np.dot(HX,v[n:n + p])),(np.dot(HX.T, v[0:n]))), axis=0)
        # else:
        #    def mv(v):
        #        return np.concatenate(((np.dot(HZ, np.dot(HZ.T, v[0:n])) +
        # np.multiply(np.multiply(alpha[i_best], R.reshape(v[0:n].shape)))
        # + np.dot(HX, v[n:n + p])),(np.dot(HX.T, v[0:n]))), axis=0)

        # Benzi et al. 2005, Eq 3.5

        def invPsi(v):
            Dvec = np.divide(
                ((1.0 / alpha[i_best]) * self.Psi_sigma),
                ((1.0 / alpha[i_best]) * self.Psi_sigma + 1),
            )
            Psi_U = np.multiply((1.0 / sqrt(alpha[i_best])), self.Psi_U)
            Psi_UTv = np.dot(Psi_U.T, v)
            # TODO: remove these stupid reshape by using vectors and matrices
            alphainvRv = (
                (1.0 / alpha[i_best]) * self.cov_obs_solve(v.ravel())
            ).reshape(-1, 1)

            if Psi_UTv.ndim == 1:
                PsiDPsiTv = np.dot(
                    Psi_U,
                    np.multiply(Dvec[: Psi_U.shape[1]].reshape(Psi_UTv.shape), Psi_UTv),
                )
            elif Psi_UTv.ndim == 2:  # for invPsi(HX)
                DMat = np.tile(
                    Dvec[: Psi_U.shape[1]], (Psi_UTv.shape[1], 1)
                ).T  # n_pc by p
                PsiDPsiTv = np.dot(Psi_U, np.multiply(DMat, Psi_UTv))
            else:
                raise ValueError(
                    "Psi_U times vector should have a dimension "
                    "smaller than 2 - current dim = %d" % (Psi_UTv.ndim)
                )

            return alphainvRv - PsiDPsiTv

        # Direct Inverse of cokkring matrix - Lee et al. WRR 2016 Eq (14)
        # typo in Eq (14), (2,2) block matrix should be -S^-1 instead of -S
        def Pmv(v):
            invPsiv = invPsi(v[0:n])
            S = np.dot(HX.T, invPsi(HX))  # p by p matrix
            invSHXTinvPsiv = np.linalg.solve(S, np.dot(HX.T, invPsiv))
            invPsiHXinvSHXTinvPsiv = invPsi(np.dot(HX, invSHXTinvPsiv))
            return np.concatenate(
                ((invPsiv - invPsiHXinvSHXTinvPsiv), (invSHXTinvPsiv)), axis=0
            )

        P = LinearOperator((n + p, n + p), matvec=Pmv, rmatvec=Pmv, dtype="d")

        ## Matrix handle for iterative approach without approximation
        # - this should be included as an option
        # n_pc = self.n_pc
        # Afun = LinearOperator((n + p, n + p), matvec=mv, rmatvec=mv, dtype='d')
        # callback = Residual()
        ## Residual and maximum iterations
        # itertol = 1.e-10 if not 'iterative_tol'
        # in self.params else self.params['iterative_tol']
        # solver_maxiter = m if not 'iterative_maxiter'
        # in self.params else self.params['iterative_maxiter']

        # start = time()
        v = np.zeros((m), dtype="d")

        for i in range(m):
            b = np.zeros((n + p, 1), dtype="d")
            b[0:n] = np.dot(
                HZ,
                (
                    np.multiply(
                        np.sqrt(self.Q.eig_vals), self.Q.eig_vects[i : i + 1, :].T
                    )
                ),
            )
            b[n : n + p] = self.drift.mat[i : i + 1, :].T

            tmp = float(np.dot(b.T, P(b))[0, 0])
            v[i] = self.prior_s_var[i] - tmp

            if i % 10000 == 0 and i > 0:
                self.loginfo("%d-th element evaluation done.." % (i))
        v = np.where(v > self.prior_s_var, self.prior_s_var, v)

        # self.loginfo("Pv compute variance: %f sec" % (time() - start))
        # self.loginfo("norm(v-v1): %g" % (np.linalg.norm(v - v1)))
        # self.loginfo("max(v-v1): %g, %g" % ((v - v1).max(),(v-v1).min()))

        return v.reshape(-1, 1)
