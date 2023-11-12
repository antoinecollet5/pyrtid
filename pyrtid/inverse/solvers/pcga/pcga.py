"""
Implement the Principal Component Geostatistical Approach for large-scale inversion.

The original code has been written by Jonghyun Harry Lee.

See: https://github.com/jonghyunharrylee/pyPCGA
"""
import multiprocessing
from dataclasses import dataclass, field
from math import isnan, sqrt
from time import time
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import scipy as sp
from scipy._lib._util import check_random_state  # To handle random_state
from scipy.sparse.linalg import (
    LinearOperator,
    eigsh,
    gmres,
    minres,
    svds,
)

from pyrtid.inverse.regularization import (
    ConstantDriftMatrix,
    DriftMatrix,
    EigenFactorizedCovarianceMatrix,
)
from pyrtid.utils import NDArrayFloat, StrEnum

__all__ = ["PCGA"]
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
class IntervalState:
    """Class to keep track of internal state."""

    # keep track of some values (best, init)
    s_best = None
    beta_best = None
    simul_obs_best = None
    iter_best = 0
    obj_best = 1.0e20
    simul_obs_init = None
    objvals: List[float] = field(default_factory=lambda: [])
    Q2_cur = 0.0
    cR_cur = 0.0
    Q2_best = 0.0
    cR_best = 0.0
    i_best = 0


class PCGA:
    """
    Solve inverse problem with PCGA (approx to quasi-linear method)

    every values are represented as 2D np array
    """

    def __init__(
        self,
        s_init: NDArrayFloat,
        obs: NDArrayFloat,
        cov_obs: NDArrayFloat,
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
        restol: float = 1e-2,
        is_post_cov: bool = False,
        is_verbose: bool = True,
        is_save_jac: bool = False,
        # PCGA parameters (purturbation size)
        precision=1.0e-8,
    ) -> None:
        """

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
        restol : float, optional
            _description_, by default 1e-2
        is_post_cov : bool, optional
            _description_, by default False
        is_verbose : bool, optional
            _description_, by default True
        is_save_jac : bool, optional
            _description_, by default False
        precision : _type_, optional
            _description_, by default 1.0e-8
        """
        ##### Forward Model
        # Make sure the array has a second dimension of length 1.
        self.s_init = np.array(s_init).reshape(-1, 1)
        # Observations
        self.obs = np.array(obs).reshape(-1, 1)
        self.cov_obs = cov_obs
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
        self.restol = restol
        self.is_post_cov: bool = is_post_cov
        self.is_verbose: bool = is_verbose
        self.post_cov_estimation: Optional[PostCovEstimation] = post_cov_estimation
        # Switch to direct if direct solve:
        # Otherwise the preconditioner is not build while it is required
        # for the diagonal post covariance estimation
        if self.post_cov_estimation is not None and self.is_direct_solve:
            self.post_cov_estimation = PostCovEstimation.DIRECT

        # PCGA parameters (purturbation size)
        self.precision: float = precision

        # TODO: parametrize
        self.nopts_lm = 4

        # keep track of the internal state
        self.istate = IntervalState()
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

        # TODO: see if we move these internal states
        self.HX = None
        self.HZ = None
        self.Hs = None
        self.P = None
        self.Psi_U = None
        self.Psi_sigma = None

        ##### Optimization
        self.display_init_parameters()

    def display_init_parameters(self) -> None:
        print("##### PCGA Inversion #####")
        print("##### 1. Initialize forward and inversion parameters")
        print("------------ Inversion Parameters -------------------------")
        _dict = {
            "Number of unknowns": self.s_dim,
            "Number of observations": self.d_dim,
            "Number of principal components (n_pc)": self.Q.n_pc,
            "Maximum Gauss-Newton iterations": self.maxiter,
            "Machine precision (delta = sqrt(precision))": self.precision,
            "Tol for iterations (norm(sol_diff)/norm(sol))": self.restol,
            "Levenberg-Marquardt (is_lm)": self.is_lm,
            "Posterior covariance computation": self.post_cov_estimation,
        }
        if self.is_lm:
            _dict["LM solution min (lm_smin)"] = self.lm_smin
            _dict["LM solution max (lm_smax)"] = self.lm_smax

        _dict["Line search"] = self.is_line_search

        # dipslay the dict content
        # first get the max length
        max_length: int = int(np.max([len(_str) for _str in _dict.keys()]))
        for k, v in _dict.items():
            print(f"  {k: <{max_length}} : {v}")

        print("-----------------------------------------------------------")

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
            "cov_obs must be a 2D matrix with "
            f"dimensions ({self.d_dim}, {self.d_dim})."
        )
        if len(cov.shape) > 2:
            raise error
        if cov.shape[0] != self.obs.size:  # type: ignore
            raise error
        if cov.ndim == 2:
            if cov.shape[0] != cov.shape[1]:  # type: ignore
                raise error

        # From iterative_ensemble_smoother code
        # Only compute the covariance factorization once
        # If it's a full matrix, we gain speedup by only computing cholesky once
        # If it's a diagonal, we gain speedup by never having to compute cholesky
        if cov.ndim == 2:
            self.cov_obs_cholesky: NDArrayFloat = sp.linalg.cholesky(cov, lower=False)
        else:
            self.cov_obs_cholesky = np.sqrt(cov)  # type: ignore

        # For now only 1D arrays are supported
        self._cov_obs: NDArrayFloat = cov.reshape(-1, 1)

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

    @property
    def sqrtR(self) -> NDArrayFloat:
        return np.sqrt(self.cov_obs)

    @property
    def invsqrtR(self) -> NDArrayFloat:
        return 1.0 / np.sqrt(self.cov_obs)

    @property
    def invR(self) -> NDArrayFloat:
        return 1.0 / self.cov_obs

    def get_v0(self, size) -> Optional[NDArrayFloat]:
        if self.random_state is not None:
            return self.random_state.uniform(size=(size,))
        else:
            return None

    def jac_vect(self, x, s, simul_obs, precision, delta=None):
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
                    * sqrt(precision)
                    * (max(abs(mag), absmag))
                    / ((np.linalg.norm(x[:, i : i + 1]) + np.finfo(float).eps) ** 2)
                )

                if deltas[i] == 0:  # s = 0 or x = 0
                    print(
                        "%d-th delta: signmag %g, precision %g, max abs %g, norm %g"
                        % (
                            i,
                            signmag,
                            precision,
                            (max(abs(mag), absmag)),
                            (np.linalg.norm(x) ** 2),
                        )
                    )

                    deltas[i] = sqrt(precision)

                    print("%d-th delta: assigned as sqrt(precision) - %g", deltas[i])
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

    def objective_function(self, s_cur, beta_cur, simul_obs) -> float:
        """
        0.5(y-h(s))^TR^{-1}(y-h(s)) + 0.5*(s-Xb)^TQ^{-1}(s-Xb)
        """
        if simul_obs is None:
            simul_obs = self.forward_model(s_cur)

        # TODO: replace with prior for the obj fun
        smxb = s_cur - np.dot(self.drift.mat, beta_cur)
        ymhs = self.obs - simul_obs

        invQs = self.Q.solve(smxb)
        # TODO: replace with solve
        obj = 0.5 * np.dot(ymhs.T, np.divide(ymhs, self.cov_obs)) + 0.5 * np.dot(
            smxb.T, invQs
        )
        return obj

    def objective_function_no_beta(self, s_cur, simul_obs) -> float:
        """
        marginalized objective w.r.t. beta
        0.5(y-h(s))^TR^{-1}(y-h(s)) + 0.5*(s-Xb)^TQ^{-1}(s-Xb)
        """
        if simul_obs is None:
            simul_obs = self.forward_model(s_cur)

        X = self.drift.mat
        self.drift.beta_dim

        ymhs = self.obs - simul_obs

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
        obj = 0.5 * np.dot(ymhs.T, np.divide(ymhs, self.cov_obs)) + 0.5 * (
            np.dot(invZs.T, invZs) - np.dot(XTinvQs.T, tmp)
        )
        return obj

    def jac_mat(self, s_cur, simul_obs, Z):
        m: int = self.s_dim
        p: int = self.drift.beta_dim
        n_pc: int = self.Q.n_pc
        precision: float = self.precision

        temp = np.zeros((m, p + n_pc + 1), dtype="d")  # [HX, HZ, Hs]

        temp[:, 0:p] = np.copy(self.drift.mat)
        temp[:, p : p + n_pc] = np.copy(Z)
        temp[:, p + n_pc : p + n_pc + 1] = np.copy(s_cur)

        Htemp = self.jac_vect(temp, s_cur, simul_obs, precision)

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

    def direct_solve(self, s_cur, simul_obs=None):
        """
        Solve the geostatistical system using a direct solver.
        Not to be used unless the number of measurements are small O(100)
        """
        print("use direct solver for saddle-point (cokrigging) system")
        m = self.s_dim
        n = self.d_dim
        p = self.drift.beta_dim
        n_pc = self.Q.n_pc
        R = self.cov_obs

        if simul_obs is None:
            simul_obs = self.forward_model(s_cur)

        Z = np.zeros((m, self.Q.n_pc), dtype="d")
        for i in range(self.Q.n_pc):
            Z[:, i : i + 1] = np.dot(
                sqrt(self.Q.eig_vals[i]), self.Q.eig_vects[:, i : i + 1]
            )  # use sqrt to make it scalar

        # Compute Jacobian-Matrix products
        start1 = time()
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
        sqrtGDCovfun = LinearOperator((n, n_pc), matvec=mv, rmatvec=rmv, dtype="d")
        # sigma_cR = svds(sqrtGDCovfun, k=min(n - p - 1, n_pc - 1), which='LM',
        # maxiter=n, return_singular_vectors=False)

        if n_pc <= n - p:
            sigma_cR = svds(
                sqrtGDCovfun,
                k=n_pc - 1,
                which="LM",
                maxiter=n - p,
                return_singular_vectors=False,
                random_state=self.random_state,
            )
        else:
            sigma_cR = svds(
                sqrtGDCovfun,
                k=n - p,
                which="LM",
                maxiter=n_pc,
                return_singular_vectors=False,
                random_state=self.random_state,
            )

        print("computed Jacobian-Matrix products in : %f secs" % (start1 - start2))
        # print("computed Jacobian-Matrix products in : %f secs, eig. val.
        # of generalized data covariance : %f secs" % (start1 - start2,time()-start2))

        # Construct HQ directly
        HQ = np.dot(HZ, Z.T)

        if self.is_lm:
            print(
                "Solve geostatistical inversion problem (co-kriging, "
                "saddle point systems) with Levenberg-Marquardt"
            )
            nopts = self.nopts_lm
            alpha = 10 ** (np.linspace(0.0, np.log10(self.alphamax_lm), nopts))

        else:
            print(
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
            Psi = np.dot(HZ, HZ.T) + np.multiply(
                np.multiply(alpha[i], R), np.eye(n, dtype="d")
            )

            # Create matrix system and solve it
            # cokriging matrix
            A = np.zeros((n + p, n + p), dtype="d")
            b = np.zeros((n + p, 1), dtype="d")

            A[0:n, 0:n] = np.copy(Psi)
            A[0:n, n : n + p] = np.copy(HX)
            A[n : n + p, 0:n] = np.copy(HX.T)

            # Ax = b, b = obs - h(s) + Hs
            b[:n] = self.obs[:] - simul_obs + Hs[:]

            x = np.linalg.solve(A, b)

            ##Extract components and return final solution
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

            Q2_all[:, i : i + 1] = np.dot(b[:n].T, xi) / (n - p)

            tmp_cR = np.zeros((n - p, 1), "d")

            if R.shape[0] == 1:
                tmp_cR[:] = np.multiply(alpha[i], self.cov_obs)
                tmp_cR[: sigma_cR.shape[0]] = (
                    tmp_cR[: sigma_cR.shape[0]] + (sigma_cR[:, np.newaxis]) ** 2
                )
            else:  # need to fix this part later 12/7/2020
                tmp_cR[:] = np.multiply(alpha[i], R[:-p])

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
            print("evaluate LM solutions")
            simul_obs_all = -10000.0 * np.ones((n, nopts), "d")
            s_hat_all_tmp = s_hat_all[:, np.invert(LM_eval)]
            simul_obs_all_tmp = self.forward_model(s_hat_all_tmp)
            simul_obs_all[:, np.invert(LM_eval)] = simul_obs_all_tmp
        else:
            print("evaluate the best solution")
            simul_obs_all = self.forward_model(s_hat_all)

        if np.size(simul_obs_all, 1) != nopts:
            raise ValueError("np.size(simul_obs_all,1) != nopts")

        obj_best = 1.0e20
        if self.is_verbose:
            print("%d objective value evaluations" % nopts)
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
                if self.is_verbose:
                    print("%d-th solution obj %e (alpha %f)" % (i, obj, alpha[i]))
                s_hat = s_hat_all[:, i : i + 1]
                beta = beta_all[:, i : i + 1]
                simul_obs_new = simul_obs_all[:, i : i + 1]
                obj_best = obj
                self.istate.Q2_cur = Q2_all[:, i : i + 1]
                self.istate.cR_cur = cR_all[:, i : i + 1]

        return s_hat, beta, simul_obs_new

    def iterative_solve(self, s_cur, simul_obs=None, precond=False):
        """
        Iterative Solve
        """
        m = self.s_dim
        n = self.d_dim
        p = self.drift.beta_dim
        n_pc = self.Q.n_pc
        R = self.cov_obs

        Z = np.zeros((m, n_pc), dtype="d")
        for i in range(n_pc):
            Z[:, i : i + 1] = np.dot(
                sqrt(self.Q.eig_vals[i]), self.Q.eig_vects[:, i : i + 1]
            )  # use sqrt to make it scalar

        if simul_obs is None:
            simul_obs = self.forward_model(s_cur)

        # Compute Jacobian-Matrix products
        start1 = time()
        HX, HZ, Hs, U_data = self.jac_mat(s_cur, simul_obs, Z)
        # debug_here()

        start2 = time()
        print("computed Jacobian-Matrix products in %f secs" % (start2 - start1))

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
            sqrtGDCovfun = LinearOperator((n, n_pc), matvec=mv, rmatvec=rmv, dtype="d")

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

            if self.is_verbose:
                print(
                    "eig. val. of generalized data covariance : "
                    "%f secs (%8.2e, %8.2e, %8.2e)"
                    % (time() - start2, sigma_cR[0], sigma_cR.min(), sigma_cR.max())
                )
            # print("computed Jacobian-Matrix products in %f secs, eig. val.
            # of generalized data covariance : %f secs (%8.2e, %8.2e, %8.2e)"
            # % (start2 - start1, time()-start2,sigma_cR[0],sigma_cR.min(),
            # sigma_cR.max()))
        else:  # Compute eig(P*(HQHT+R)*P) approximately by svd(P*(HZ*HZ' + R)*P)
            # need to do for each alpha[i]*R
            pass
            # print("computed Jacobian-Matrix products in %f secs" % (start2 - start1))

        # preconditioner construction
        # will add more description here
        if self.is_use_preconditioner:
            tStart_precond = time()
            # GHEP : HQHT u = lamdba R u => u = R^{-1/2} y
            if R.shape[0] == 1:
                # original implementation was sqrt of R^{-1/2} HZ n by n_pc
                # svds cannot compute entire n_pc eigenvalues so do this for
                # n by n matrix
                # this leads to double the cost
                def pmv(v):
                    return np.multiply(
                        self.invsqrtR,
                        np.dot(HZ, (np.dot(HZ.T, np.multiply(self.invsqrtR, v)))),
                    )
                    # return np.multiply(self.invsqrtR,np.dot(HZ,v))

                def prmv(v):
                    # return np.dot(HZ.T,np.multiply(self.invsqrtR,v))
                    return pmv(v)

            else:
                # n by n
                def pmv(v):
                    return np.multiply(
                        self.invsqrtR.reshape(v.shape),
                        np.dot(
                            HZ,
                            (
                                np.dot(
                                    HZ.T, np.multiply(self.invsqrtR.reshape(v.shape), v)
                                )
                            ),
                        ),
                    )

                def prmv(v):
                    return pmv(v)

            # if self.is_verbose:
            #    print('preconditioner construction using Generalized
            # Eigen-decomposition')
            #    print("n :%d & n_pc: %d" % (n,n_pc))

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

            # print("eig. val. of sqrt data covariance (%8.2e, %8.2e, %8.2e)"
            # % (Psi_sigma[0], Psi_sigma.min(), Psi_sigma.max()))
            # print(Psi_sigma)

            Psi_U = np.multiply(self.invsqrtR, Psi_U)
            # if R.shape[0] == 1:
            # Psi_sigma = Psi_sigma**2 # because we use svd(HZ)
            # instead of svd(HQHT+R)
            index_Psi_sigma = np.argsort(Psi_sigma)
            index_Psi_sigma = index_Psi_sigma[::-1]
            Psi_sigma = Psi_sigma[index_Psi_sigma]
            Psi_U = Psi_U[:, index_Psi_sigma]
            Psi_U = Psi_U[:, Psi_sigma > 0]
            Psi_sigma = Psi_sigma[Psi_sigma > 0]

            if self.is_verbose:
                print(
                    "time for data covarance construction : %f sec "
                    % (time() - tStart_precond)
                )
                print(
                    "eig. val. of data covariance (%8.2e, %8.2e, %8.2e)"
                    % (Psi_sigma[0], Psi_sigma.min(), Psi_sigma.max())
                )
                if Psi_U.shape[1] != n_pc:
                    print(
                        "- rank of data covariance :%d for preconditioner construction"
                        % (Psi_U.shape[1])
                    )

            self.Psi_sigma = Psi_sigma
            self.Psi_U = Psi_U

        if self.is_lm:
            print("solve saddle point (co-kriging) systems with Levenberg-Marquardt")
            nopts = self.nopts_lm
            alpha = 10 ** (np.linspace(0.0, np.log10(self.alphamax_lm), nopts))
        else:
            print("solve saddle point (co-kriging) system")
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

                def mv(v):
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

                def mv(v):
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
            Afun = LinearOperator((n + p, n + p), matvec=mv, rmatvec=mv, dtype="d")

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

                        alphainvRv = np.multiply(
                            np.multiply((1.0 / alpha[i]), self.invR), v
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

                        if Psi_UTv.ndim == 1:
                            alphainvRv = np.multiply(
                                np.multiply(
                                    (1.0 / alpha[i]), self.invR.reshape(v.shape)
                                ),
                                v,
                            )
                            PsiDPsiTv = np.dot(
                                Psi_U_i,
                                np.multiply(
                                    Dvec[: Psi_U_i.shape[1]].reshape(Psi_UTv.shape),
                                    Psi_UTv,
                                ),
                            )
                        elif Psi_UTv.ndim == 2:  # for invPsi(HX)
                            # 14/06/2018 Harry
                            # may need to change this later in a more general way
                            RMat = np.tile(
                                np.multiply((1.0 / alpha[i]), self.invR),
                                Psi_UTv.shape[1],
                            )
                            alphainvRv = np.multiply(RMat, v)
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

                P = LinearOperator((n + p, n + p), matvec=Pmv, rmatvec=Pmv, dtype="d")

                # TODO: parametrize
                restart = 50
                x, info = gmres(
                    Afun,
                    b,
                    tol=itertol,
                    restart=restart,
                    maxiter=solver_maxiter,
                    callback=callback,
                    M=P,
                )
                if self.is_verbose:
                    print(
                        "-- Number of iterations for gmres %g" % (callback.itercount())
                    )
                if info != 0:  # if not converged
                    callback = Residual()
                    x, info = minres(
                        Afun,
                        b,
                        x0=x,
                        tol=itertol,
                        maxiter=solver_maxiter,
                        callback=callback,
                        M=P,
                    )
                    if self.is_verbose:
                        print(
                            "-- Number of iterations for minres %g and info %d"
                            % (callback.itercount(), info)
                        )
            else:
                x, info = minres(
                    Afun, b, tol=itertol, maxiter=solver_maxiter, callback=callback
                )
                if self.is_verbose:
                    print(
                        "-- Number of iterations for minres %g" % (callback.itercount())
                    )

                if info != 0:
                    x, info = gmres(
                        Afun,
                        b,
                        x0=x,
                        tol=itertol,
                        maxiter=solver_maxiter,
                        callback=callback,
                    )
                    print(
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

            if self.is_verbose:
                if LM_eval[i]:
                    print(
                        "%d - min(s): %g, max(s) :%g - violate LM_smin or LM_smax"
                        % (
                            i,
                            s_hat_all[:, i : i + 1].min(),
                            s_hat_all[:, i : i + 1].max(),
                        )
                    )
                else:
                    print(
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
                tmp_cR[:] = np.multiply(alpha[i], R[:-p])

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
            print("evaluate LM solutions")
            simul_obs_all = -10000.0 * np.ones((n, nopts), "d")
            s_hat_all_tmp = s_hat_all[:, np.invert(LM_eval)]
            simul_obs_all_tmp = self.forward_model(s_hat_all_tmp)
            print("LM solution evaluated")
            simul_obs_all[:, np.invert(LM_eval)] = simul_obs_all_tmp

        else:
            print("evaluate the best solution")
            simul_obs_all = self.forward_model(s_hat_all)

        if np.size(simul_obs_all, 1) != nopts:
            return ValueError("np.size(simul_obs_all,1) should be nopts")

        # evaluate objective values and select best value
        obj_best = 1.0e20
        if self.is_lm and self.is_verbose:
            print("%d objective value evaluations" % nopts)

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
                self.istate.Q2_cur = Q2_all[:, i : i + 1]
                self.istate.cR_cur = cR_all[:, i : i + 1]
                i_best = i
                if self.is_verbose:
                    print(
                        "{:d}-th solution obj {} (alpha {}, beta {})".format(
                            i, obj.reshape(-1), alpha[i], beta.reshape(-1).tolist()
                        )
                    )

        if i_best == -1:
            print("no better solution found ..")
            s_hat = s_cur
            simul_obs_new = simul_obs
            beta = 0.0

        if self.post_cov_estimation is not None:
            self.HZ = HZ
            self.HX = HX
            self.R_LM = np.multiply(alpha[i_best], self.cov_obs)

        self.istate.i_best = i_best  # keep track of best LM solution
        return s_hat, beta, simul_obs_new

    def linear_iteration(self, s_cur, simul_obs):
        # Solve geostatistical system -> two ways, direct of iterative solve
        if self.is_direct_solve:
            s_hat, beta, simul_obs_new = self.direct_solve(s_cur, simul_obs)
        else:
            s_hat, beta, simul_obs_new = self.iterative_solve(
                s_cur, simul_obs, precond=self.is_use_preconditioner
            )
        obj = self.objective_function(s_hat, beta, simul_obs_new)

        return s_hat, beta, simul_obs_new, obj

    def line_search(self, s_cur, s_past):
        nopts = self.nopts_lm
        m = self.s_dim

        s_hat_all = np.zeros((m, nopts), "d")
        # need to remove delta = 0 and 1
        delta = np.linspace(-0.1, 1.1, nopts)

        for i in range(nopts):
            s_hat_all[:, i : i + 1] = delta[i] * s_past + (1.0 - delta[i]) * s_cur

        print("evaluate linesearch solutions")
        simul_obs_all = self.forward_model(s_hat_all)

        # will change assert to valueerror
        assert np.size(simul_obs_all, 1) == nopts
        obj_best = 1.0e20

        for i in range(nopts):
            obj = self.objective_function_no_beta(
                s_hat_all[:, i : i + 1], simul_obs_all[:, i : i + 1]
            )

            if obj < obj_best:
                if self.is_verbose:
                    print("%d-th solution obj %e (delta %f)" % (i, obj, delta[i]))
                s_hat = s_hat_all[:, i : i + 1]
                simul_obs_new = simul_obs_all[:, i : i + 1]
                obj_best = obj

        return s_hat, simul_obs_new, obj_best

    def gauss_newton(self):
        """
        Gauss-newton iteration
        """

        s_init = self.s_init
        self.maxiter
        restol = self.restol

        obj = VERY_LARGE_NUMBER

        res = 1.0

        print("##### 2. Start PCGA Inversion #####")
        print("-- evaluate initial solution")

        # s_init has shape (s_dim, 1)
        simul_obs_init = self.forward_model(s_init)

        self.simul_obs_init = simul_obs_init
        residuals = simul_obs_init - self.obs
        RMSE_init = np.linalg.norm(residuals) / np.sqrt(self.d_dim)
        nRMSE_init = np.linalg.norm(np.divide(residuals, self.sqrtR)) / np.sqrt(
            self.d_dim
        )
        print(
            f"- obs. RMSE (norm(obs. diff.)/sqrt(nobs)): {RMSE_init}\n"
            f"- normalized obs. RMSE (norm(obs. diff./sqrtR)/sqrt(nobs)): {nRMSE_init}"
        )

        simul_obs = np.copy(simul_obs_init)
        s_cur = np.copy(s_init)
        s_past = np.copy(s_init)

        for i in range(self.maxiter):  # type: ignore
            start = time()

            # TODO: make a loop for that

            print(f"***** Iteration {i + 1} ******")
            s_cur, beta_cur, simul_obs_cur, obj = self.linear_iteration(
                s_past, simul_obs
            )

            print(
                "- Geostat. inversion at iteration %d is %g sec"
                % ((i + 1), round(time() - start))
            )

            if obj < self.istate.obj_best:
                self.istate.obj_best = obj
                self.istate.s_best = s_cur
                self.istate.beta_best = beta_cur
                self.istate.simul_obs_init = simul_obs_cur
                self.istate.iter_best = i + 1
                self.istate.Q2_best = self.istate.Q2_cur
                self.istate.cR_best = self.istate.cR_cur
            else:
                if self.is_line_search:
                    print("perform simple linesearch due to no progress in obj value")
                    s_cur, simul_obs_cur, obj = self.line_search(s_cur, s_past)
                    if obj < self.istate.iter_best:
                        self.istate.obj_best = obj
                        self.istate.s_best = s_cur
                        self.istate.simul_obs_init = simul_obs_cur
                        self.istate.iter_best = i + 1
                    else:
                        if i > 1:
                            print("no progress in obj value")
                            i + 1
                            break
                        else:
                            print(
                                "no progress in obj value but wait for one "
                                "more iteration.."
                            )
                            # allow first few iterations
                            pass  # allow for
                else:
                    print("no progress in obj value")
                    i + 1
                    break

            res = np.linalg.norm(s_past - s_cur) / np.linalg.norm(s_past)
            RMSE_cur = np.linalg.norm(simul_obs_cur - self.obs) / np.sqrt(self.d_dim)
            nRMSE_cur = np.linalg.norm(
                np.divide(simul_obs_cur - self.obs, self.sqrtR)
            ) / np.sqrt(self.d_dim)

            print("== iteration %d summary ==" % (i + 1))
            print(
                "= objective function is %e, relative L2-norm diff btw "
                "sol %d and sol %d is %g" % (obj, i, i + 1, res)
            )
            print(
                "= obs. RMSE is %g, obs. normalized RMSE is %g" % (RMSE_cur, nRMSE_cur)
            )

            self.istate.objvals.append(float(obj))

            if res < restol:
                i + 1
                break

            s_past = np.copy(s_cur)
            simul_obs = np.copy(simul_obs_cur)

        if self.post_cov_estimation is not None:
            # assume linesearch result close to the current solution
            start = time()
            if self.istate.i_best is None:
                self.istate.i_best = 0
            if self.post_cov_estimation == PostCovEstimation.DIRECT:
                print(
                    "start direct posterior variance computation "
                    "- this option works for O(nobs) ~ 100"
                )
                self.post_diagv = self.ComputePosteriorDiagonalEntriesDirect(
                    self.HZ, self.HX, self.istate.i_best, self.cov_obs
                )
            else:
                print("start posterior variance computation")
                self.post_diagv = self.ComputePosteriorDiagonalEntries(
                    self.HZ, self.HX, self.istate.i_best, self.cov_obs
                )
            print("posterior diag. computed in %f secs" % (time() - start))
            # if self.iter_save:
            #     np.savetxt("./postv.txt", self.post_diagv)

        # return s_cur, beta_cur, simul_obs, iter_cur
        print("------------ Inversion Summary ---------------------------")
        print(f"** Found solution at iteration {self.istate.iter_best}")
        print(
            "** Solution obs. RMSE %g , initial obs. RMSE %g, "
            "where RMSE = (norm(obs. diff.)/sqrt(nobs)), "
            "Solution obs. nRMSE %g, init. obs. nRMSE %g"
            % (
                np.linalg.norm(self.istate.simul_obs_init - self.obs)
                / np.sqrt(self.d_dim),
                RMSE_init,
                np.linalg.norm(
                    np.divide(self.istate.simul_obs_init - self.obs, self.sqrtR)
                )
                / np.sqrt(self.d_dim),
                nRMSE_init,
            )
        )
        print(f"** Final objective function value is {self.istate.obj_best}")
        print(
            "** Final predictive model checking Q2, cR is %e, %e"
            % (self.istate.Q2_best, self.istate.cR_best)
        )

        return (
            self.istate.s_best,
            self.istate.simul_obs_init,
            self.post_diagv,
            self.istate.iter_best,
        )

    def run(self) -> Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, int]:
        start = time()
        s_hat, simul_obs, post_diagv, iter_best = self.gauss_newton()
        print(f"** Total elapsed time is {(time() - start)} secs")
        print("----------------------------------------------------------")
        if self.post_cov_estimation is not None:
            return s_hat, simul_obs, post_diagv, iter_best
        else:
            return s_hat, simul_obs, iter_best

    def ComputePosteriorDiagonalEntriesDirect(self, HZ, HX, i_best, R):
        """
        Computing posterior diagonal entries
        Don't use this for large number of measurements!
        Works best for small measurements O(100)
        """

        m = self.s_dim
        n = self.d_dim
        p = self.drift.beta_dim

        alpha = 10 ** (np.linspace(0.0, np.log10(self.alphamax_lm), self.nopts_lm))
        Ri = np.multiply(alpha[i_best], R)

        n_pc = self.Q.n_pc
        Z = np.zeros((m, n_pc), dtype="d")
        for i in range(n_pc):
            Z[:, i : i + 1] = np.dot(
                sqrt(self.Q.eig_vals[i]), self.Q.eig_vects[:, i : i + 1]
            )  # use sqrt to make it scalar

        v = np.zeros((m, 1), dtype="d")

        # Construct Psi directly
        if isinstance(Ri, float):
            Psi = np.dot(HZ, HZ.T) + np.multiply(Ri, np.eye(n, dtype="d"))
        elif Ri.shape[0] == 1 and Ri.ndim == 1:
            Psi = np.dot(HZ, HZ.T) + np.multiply(Ri, np.eye(n, dtype="d"))
        else:
            Psi = np.dot(HZ, HZ.T) + np.diag(
                Ri.reshape(-1)
            )  # reshape Ri from (n,1) to (n,) for np.diag

        HQ = np.dot(HZ, Z.T)

        # Create matrix system and solve it
        # cokriging matrix
        A = np.zeros((n + p, n + p), dtype="d")
        b = np.zeros((n + p, 1), dtype="d")

        A[0:n, 0:n] = np.copy(Psi)
        A[0:n, n : n + p] = np.copy(HX)
        A[n : n + p, 0:n] = np.copy(HX.T)

        # HQX = np.vstack((HQ,self.drift.mat.T))
        # diagred = np.diag(np.dot(HQX.T, np.linalg.solve(A, HQX)))
        # diagred1 = np.diag(np.dot(HQ.T, np.linalg.solve(Psi, HQ)))
        # HQX1 = np.vstack((HQ,self.drift.mat[:,0].T))
        # A1 = np.zeros((n+1,n+1),dtype='d')
        # A1[0:n,0:n] = np.copy(Psi)
        # A1[0:n,n:n+1] = np.copy(HX[:,0:1])
        # A1[n:n+1,0:n] = np.copy(HX[:,0:1].T)

        # diagred2 = np.diag(np.dot(HQX1.T, np.linalg.solve(A1, HQX1)))
        # v1 = priorvar - diagred

        print("We have reached that point !")

        for i in range(m):
            b = np.zeros((n + p, 1), dtype="d")
            b[0:n] = HQ[:, i : i + 1]
            b[n : n + p] = self.drift.mat[i : i + 1, :].T
            tmp = np.dot(b.T, np.linalg.solve(A, b))
            v[i] = self.prior_s_var[i] - tmp
            # if v[i] <= 0:
            #    print("%d-th element negative" % (i))
            if i % 1000 == 0:
                print("%d-th element evaluated" % (i))

        # print("compute variance: %f sec" % (time() - start))
        return v

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

            if R.shape[0] == 1:
                alphainvRv = np.multiply(
                    np.multiply((1.0 / alpha[i_best]), self.invR), v
                )
            elif Psi_UTv.ndim == 1:
                alphainvRv = np.multiply(
                    np.multiply((1.0 / alpha[i_best]), self.invR.reshape(v.shape)),
                    v,
                )
            else:
                RMat = np.tile(
                    np.multiply((1.0 / alpha[i_best]), self.invR), Psi_UTv.shape[1]
                )
                alphainvRv = np.multiply(RMat, v)

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

            # invAb, info = gmres(Afun, b, tol=itertol,
            # maxiter=solver_maxiter, callback=callback, M=P)
            ##invAb, info = minres(Afun, b, tol=itertol,
            # maxiter=solver_maxiter, callback=callback, M=P)

            v[i] = self.prior_s_var[i] - np.dot(b.T, P(b))

            # if i < 15:
            #    tmp = np.dot(b.T, np.linalg.solve(A, b))
            #    callback = Residual()
            #    invAb, info = gmres(Afun, b, tol=itertol,
            # maxiter=solver_maxiter, callback=callback, M=P)
            #    print("-- Number of iterations for gmres %g
            # and info %d" % (callback.itercount(), info))
            #    print("%d: %g %g %g" % (i, v[i], priorvar
            # - np.dot(b.T, invAb.reshape(-1,1)),priorvar - tmp))

            if i % 10000 == 0 and i > 0 and self.is_verbose:
                print("%d-th element evaluation done.." % (i))
        v = np.where(v > self.prior_s_var, self.prior_s_var, v)

        # print("Pv compute variance: %f sec" % (time() - start))
        # print("norm(v-v1): %g" % (np.linalg.norm(v - v1)))
        # print("max(v-v1): %g, %g" % ((v - v1).max(),(v-v1).min()))

        return v.reshape(-1, 1)
