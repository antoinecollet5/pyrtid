"""
Implement the interface for the PCGA inversion executor.

PCGA stands for Principal Component Geostatistical Approach.

This inversion executor gives access to the PyPCGA implementation to solve the inverse
problem. The original code is provided and maintained by JONGHYUN “HARRY” LEE
and can be found at: https://github.com/jonghyunharrylee/pyPCGA


References
----------

- J Lee, H Yoon, PK Kitanidis, CJ Werth, AJ Valocchi, "Scalable subsurface inverse
modeling of huge data sets with an application to tracer concentration breakthrough data
from magnetic resonance imaging", Water Resources Research 52 (7), 5213-5231

- AK Saibaba, J Lee, PK Kitanidis, Randomized algorithms for generalized Hermitian
 eigenvalue problems with application to computing Karhunen–Loève expansion, Numerical
   Linear Algebra with Applications 23 (2), 314-339

- J Lee, PK Kitanidis, "Large‐scale hydraulic tomography and joint inversion of head
and tracer data using the Principal Component Geostatistical Approach (PCGA)",
WRR 50 (7), 5410-5427

- PK Kitanidis, J Lee, Principal Component Geostatistical Approach for large‐dimensional
inverse problems, WRR 50 (7), 5428-5443

# Applications

- T. Kadeethum, D. O'Malley, JN Fuhg, Y. Choi, J. Lee, HS Viswanathan and N. Bouklas,
A framework for data-driven solution and parameter estimation of PDEs using conditional
generative adversarial networks, Nature Computational Science, 819–829, 2021

- J Lee, H Ghorbanidehno, M Farthing, T. Hesser, EF Darve, and PK Kitanidis, Riverine
bathymetry imaging with indirect observations, Water Resources
Research, 54(5): 3704-3727, 2018

- J Lee, A Kokkinaki, PK Kitanidis, Fast large-scale joint inversion for deep aquifer
characterization using pressure and heat tracer measurements, Transport in Porous
Media, 123(3): 533-543, 2018

- PK Kang, J Lee, X Fu, S Lee, PK Kitanidis, J Ruben, Improved Characterization of
Heterogeneous Permeability in Saline Aquifers from Transient Pressure Data during
Freshwater Injection, Water Resources Research, 53(5): 4444-458, 2017

- S. Fakhreddine, J Lee, PK Kitanidis, S Fendorf, M Rolle, Imaging Geochemical
Heterogeneities Using Inverse Reactive Transport Modeling: an Example Relevant for
Characterizing Arsenic Mobilization and Distribution, Advances in
Water Resources, 88: 186-197, 2016
"""
import multiprocessing
from dataclasses import astuple, dataclass
from typing import Any, Callable, Optional, Sequence, Union

from pyrtid.inverse.executors.base import (
    BaseInversionExecutor,
    BaseSolverConfig,
    base_solver_config_params_ds,
    register_params_ds,
)
from pyrtid.inverse.regularization import DriftMatrix, EigenFactorizedCovarianceMatrix
from pyrtid.inverse.solvers import PCGA, PostCovEstimation
from pyrtid.utils.types import NDArrayFloat

pcga_solver_config_params_ds = """solver_kwargs: Optional[Dict[str, Any]]
            Additional arguments for PCGA instance. The default is None.
    """


@register_params_ds(pcga_solver_config_params_ds)
@register_params_ds(base_solver_config_params_ds)
@dataclass
class PCGASolverConfig(BaseSolverConfig):
    """
    Principal Component Geostatistical Approach Inversion Configuration.

    Attributes
    ----------
    """

    eig_cov: Optional[EigenFactorizedCovarianceMatrix] = None
    drift: Optional[DriftMatrix] = None
    prior_s_var: Optional[Union[float, NDArrayFloat]] = None
    callback: Optional[Callable] = None
    is_line_search: bool = False
    is_lm: bool = False
    is_direct_solve: bool = False
    is_use_preconditioner: bool = False
    post_cov_estimation: Optional[PostCovEstimation] = None
    is_objfun_exact: bool = False  # former objeval
    max_it_lm: int = multiprocessing.cpu_count()
    alphamax_lm: float = 10.0**3.0  # does it sound ok?
    lm_smin: Optional[float] = None
    lm_smax: Optional[float] = None
    max_it_ls: int = 20
    maxiter: int = 10
    restol: float = 1e-2
    is_post_cov: bool = False
    is_verbose: bool = True
    is_save_jac: bool = False
    eps = 1.0e-8

    def __iter__(self):
        return iter(astuple(self))


class PCGAInversionExecutor(BaseInversionExecutor[PCGASolverConfig]):
    """Principal Component Geostatistical Approach Inversion Executor."""

    def _init_solver(self, s_init: Optional[NDArrayFloat] = None) -> None:
        """Initiate a solver with its args."""
        # Array with grid coordinates. (X, Y, Z)...
        # Note: for regular grid you don't need to specify pts.
        if self.solver_config.eig_cov is not None:
            self.solver: PCGA = PCGA(
                self.data_model.s_init.ravel(),  # Must be a vector
                self.data_model.obs,
                self.data_model.cov_obs,
                self._map_forward_model_wrapper,
                self.solver_config.eig_cov,
                drift=self.solver_config.drift,
                prior_s_var=self.solver_config.prior_s_var,
                callback=self.solver_config.callback,
                is_line_search=self.solver_config.is_line_search,
                is_lm=self.solver_config.is_lm,
                is_direct_solve=self.solver_config.is_direct_solve,
                is_use_preconditioner=self.solver_config.is_use_preconditioner,
                random_state=self.solver_config.random_state,
                post_cov_estimation=self.solver_config.post_cov_estimation,
                is_objfun_exact=self.solver_config.is_objfun_exact,
                max_it_lm=self.solver_config.max_it_lm,
                alphamax_lm=self.solver_config.alphamax_lm,
                lm_smin=self.solver_config.lm_smin,
                lm_smax=self.solver_config.lm_smax,
                max_it_ls=self.solver_config.max_it_ls,
                maxiter=self.solver_config.maxiter,
                restol=self.solver_config.restol,
                is_post_cov=self.solver_config.is_post_cov,
                is_verbose=self.solver_config.is_verbose,
                is_save_jac=self.solver_config.is_save_jac,
                eps=self.solver_config.eps,
            )
        else:
            raise ValueError()

    def _get_solver_name(self) -> str:
        """Return the solver name."""
        return "PCGA"

    def run(self) -> Sequence[Any]:
        """
        Run the history matching.

        First is creates raw folders to store the different runs
        required by the HM algorithms.
        """
        super().run()
        return self.solver.run()

    def _map_forward_model_wrapper(
        self, s_ensemble: NDArrayFloat, is_parallel: bool = False, ncores: int = 1
    ) -> NDArrayFloat:
        """
        Call the forward model for all ensemble members, return predicted data.

        Function calling the non-linear observation model (forward_model)
        for all ensemble members and returning the predicted data for
        each ensemble member. this function is responsible for the creation of
        simulation folder etc.

        Returns
        -------
        None.
        """
        # pylint: disable=W0613  # Unused argument 'ncores'
        # The transposition is due to the implementation of pypcga
        return super()._map_forward_model(s_ensemble, is_parallel)
