"""
Implement the interface for the SIES inversion executor.

SIES stands for Sequential Iterative Approach.

This inversion executor gives access to the iterative_ensemble_smoother implementation
to solve the inverse problem. The original code is provided and maintained *
by Equinor and can be found at: https://github.com/equinor/iterative_ensemble_smoother

References
----------
Geir Evensen, et al. Efficient
implementation of an iterative ensemble smoother for data assimilation and reservoir
history matching. Frontiers in Applied Mathematics and Statistics, 5:47, 2019.
URL: https://www.frontiersin.org/articles/10.3389/fams.2019.00047/full.

"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional

from iterative_ensemble_smoother import SIES, steplength_exponential

from pyrtid.inverse.executors.base import (
    BaseInversionExecutor,
    BaseSolverConfig,
    base_solver_config_params_ds,
    register_params_ds,
)
from pyrtid.utils.types import NDArrayFloat


class SIESInversionType(str, Enum):
    r"""Inversion type for the computation of (S @ S.T + E @ E.T)^-1.

    Note
    ----
    It is a hashable string enum and can be iterated.

    Available inversions are:

        * `direct`:
            Solve Eqn (42) directly, which involves inverting a
            matrix of shape (num_parameters, num_parameters).
        * `subspace_exact` :
            Solve Eqn (42) using Eqn (50), i.e., the Woodbury
            lemma to invert a matrix of size (ensemble_size, ensemble_size).
            This is the method of choice when using a diagonal observation error
            covariance matrix :math:\mathbf{C}_{dd}` (also noted :math:\mathbf{R}`).
            This is always the
            case with PyRTID up to now.
        * `subspace_projected` :
            Solve Eqn (42) using Section 3.3, i.e., by projecting the covariance
            onto S. This approach utilizes the truncation factor `truncation`.
            This is the method of choice when using a full observation error
            covariance matrix :math:\mathbf{C}_{dd}`.
    """

    DIRECT = "direct"
    SUBSPACE_EXACT = "subspace_exact"
    SUBSPACE_PROJECTED = "subspace_projected"

    def __str__(self) -> str:
        """Return instance value."""
        return self.value

    def __hash__(self) -> int:
        """Return the hash of the value."""
        return hash(self.value)

    def __eq__(self, other: object) -> bool:
        """Return if two instances are equal."""
        if not isinstance(other, SIESInversionType) and not isinstance(other, str):
            return False
        return self.value == other

    @classmethod  # type: ignore
    def to_list(cls) -> List[SIESInversionType]:
        """Return all enums as a list."""
        return list(cls)


sies_solver_config_params_ds = """n_iterations : int, optional
        Number of iterations (:math:`N_{a}`). The default is 4.
    inversion_type: SIESInversionType
        Type of inversion used. See :class:`SIESInversionType` for available types.
        The default is \"subspace_exact\".
    save_ensembles_history: bool, optional
        Whether to save the history predictions and parameters over
        the assimilations. The default is False.
    truncation: float
        How much of the total energy (singular values squared) to keep in the
        SVD when `inversion` equals `subspace_projected`. Choosing 1.0
        retains all information, while 0.0 removes all information.
        The default is 0.99.
    seed: Optional[int]
        Seed for the white noise generator used in the perturbation step.
        If None, the default :func:`numpy.random.default_rng()` is used.
        The default is None.
    steplength_strategy: Optional[Callable[[int, ...], float]]
        Callable which takes the iteration number as input and returns the steplength
        for the gauss-newton iteration (between 0 and 1.0). By default it uses an
        exponential strategy: (Eq. (49), which calculates a suitable step length for
        the update step, from the book: \"Formulating the history matching problem with
        consistent error statistics", written by
        :cite:t:`evensen2021formulating`.
    is_forecast_for_last_assimilation: bool, optional
        Whether to compute the predictions for the ensemble obtained at the
        last assimilation step. The default is True.
"""


@register_params_ds(sies_solver_config_params_ds)
@register_params_ds(base_solver_config_params_ds)
@dataclass
class SIESSolverConfig(BaseSolverConfig):
    """
    Ensemble Smoother with Multiple Data Assimilation Inversion Configuration.

    Attributes
    ----------
    """

    n_iterations: int = 4
    inversion_type: SIESInversionType = SIESInversionType.SUBSPACE_EXACT
    save_ensembles_history: bool = False
    truncation: float = 0.99
    seed: Optional[int] = None
    steplength_strategy: Callable[[int], float] = steplength_exponential
    is_forecast_for_last_assimilation: bool = True


class _SIES(SIES):
    """Wrapper for the SIES class."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the instance."""
        super().__init__(*args, **kwargs)
        self.d_history: List[NDArrayFloat] = []
        self.s_history: List[NDArrayFloat] = []


class SIESInversionExecutor(BaseInversionExecutor[SIESSolverConfig]):
    """Ensemble Smoother with Multiple Data Assimilation Inversion Executor."""

    solver: _SIES

    def _init_solver(self, s_init: NDArrayFloat) -> None:
        """Initiate a solver with its args."""

        self.solver: _SIES = _SIES(
            s_init,
            self.data_model.cov_obs,
            self.data_model.obs,
            inversion=self.solver_config.inversion_type,
            truncation=self.solver_config.truncation,
            seed=self.solver_config.seed,
        )

    def _get_solver_name(self) -> str:
        """Return the solver name."""
        return "SIES"

    def run(self) -> NDArrayFloat:
        """
        Run the history matching.

        First is creates raw folders to store the different runs
        required by the HM algorithms.
        """
        super().run()
        _m = self.solver.X  # stored in the SIES instance
        if self.solver_config.save_ensembles_history:
            self.solver.s_history.append(_m)
        for iteration in range(1, self.solver_config.n_iterations + 1):  # type: ignore
            logging.info(f"Iteration # {iteration}")
            d_pred = self._map_forward_model(
                _m,
                is_parallel=self.solver_config.is_parallel,
            )
            self.solver.d_history.append(d_pred)

            _m = self.solver.sies_iteration(
                d_pred,
                step_length=self.solver_config.steplength_strategy(iteration),
            )

            if self.solver_config.save_ensembles_history:
                self.solver.s_history.append(_m)
        return _m

    @property
    def s_history(self) -> List[NDArrayFloat]:
        """Return the successive ensembles."""
        return self.solver.s_history


if __name__ == "__main__":
    # For ProcessPoolExecutor
    # Make sure that the main module can be safely imported by a new Python interpreter
    # without causing unintended side effects (such a starting a new process).
    pass
