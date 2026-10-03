# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 Antoine COLLET

"""
Provide the reactive transport solver.

The coupling between the flow, the transport and the chemistry relies on a
sequential, iterative approach with operator splitting (SIA):

1. the flow is solved once per timestep (the flow does not depend on the chemistry),
2. the transport and the chemistry are then solved alternately (*fixed point
   iterations*, FPI) until the mineral grades do not vary anymore.

If the fixed point iterations do not converge within
:attr:`~pyrtid.forward.models.TransportModel.max_fpi` iterations, the timestep is
solved again, first without the numerical acceleration (if enabled) and then with a
smaller timestep.
"""

from __future__ import annotations

import logging

import numpy as np
import scipy as sp

from pyrtid.utils import NDArrayFloat

from .flow_solver import (
    compute_u_darcy_div,
    solve_flow_stationary,
    solve_flow_transient_semi_implicit,
)
from .geochem_solver import solve_geochem
from .models import (
    H_PLUS_CONC,
    TDS_LINEAR_COEFFICIENT,
    VERY_SMALL_NUMBER,
    WATER_DENSITY,
    WATER_MW,
    FlowRegime,
    ForwardModel,
    TransportModel,
)
from .transport_solver import solve_transport_semi_implicit

logger = logging.getLogger(__name__)


class CouplingConvergenceError(RuntimeError):
    """
    Raised when the transport-chemistry coupling does not converge.

    It is raised when the fixed point iterations fail to converge for the smallest
    timestep allowed (``dt_min``) and with the numerical acceleration disabled.
    """


def get_max_coupling_error(current_arr: NDArrayFloat, prev_arr: NDArrayFloat) -> float:
    """
    Return the maximum relative variation between two successive iterates.

    Parameters
    ----------
    current_arr : NDArrayFloat
        Current iterate.
    prev_arr : NDArrayFloat
        Previous iterate, with the same shape as ``current_arr``.

    Returns
    -------
    float
        :math:`\\max |1 - x^{k+1} / x^{k}|`. Values whose magnitude is below
        ``VERY_SMALL_NUMBER`` are replaced by ``VERY_SMALL_NUMBER`` to avoid
        divisions by zero.
    """
    num = np.where(
        np.abs(current_arr) <= VERY_SMALL_NUMBER, VERY_SMALL_NUMBER, current_arr
    )
    den = np.where(np.abs(prev_arr) <= VERY_SMALL_NUMBER, VERY_SMALL_NUMBER, prev_arr)
    return float(
        np.nan_to_num(
            np.max(np.abs(1 - num / den)),
            nan=0.0,
        )
    )


def get_max_coupling_error_forward(tr_model: TransportModel, time_index: int) -> float:
    r"""
    Return the maximum transport-chemistry coupling error.

    The fixed point iteration convergence criteria reads:

    .. math::
        \text{max} \left\lVert 1 - \dfrac{\overline{c}^{n+1, k+1}}
        {\overline{c}^{n+1, k}} \right\rVert  < \epsilon

    with $k$ the number of fixed point iterations.

    This error is evaluated from the immobile concentrations (mineral grades).
    """
    return get_max_coupling_error(tr_model.limmob[time_index], tr_model.immob_prev)


class ForwardSolver:
    """
    Class solving the reactive transport forward systems.

    The solver holds no data: it operates (and stores the results) in the
    :class:`~pyrtid.forward.models.ForwardModel` given at the instantiation.
    """

    def __init__(self, model: ForwardModel) -> None:
        """
        Initialize the instance.

        Parameters
        ----------
        model : ForwardModel
            The model to solve. It is **not** copied: it is modified in place by
            :meth:`solve`. Use :func:`copy.deepcopy` beforehand to preserve it.
        """
        self.model: ForwardModel = model

    def initialize(self) -> None:
        """
        Initialize the system (t=0).

        It includes the stationary flow resolution.
        """
        # Reinit all
        self.model.reinit()

        # If stationary -> equilibrate the initial heads with sources
        # and boundary conditions

        # Get the flow and concentration sources
        unitflw_sources, conc_sources = self.model.get_sources(
            self.model.time_params.time_elapsed, self.model.grid
        )
        self.model.fl_model.lunitflow.append(unitflw_sources)
        self.model.tr_model.lsources.append(conc_sources)

        # Update the initial density
        self.model.tr_model.ldensity.append(
            get_density(
                self.model.tr_model.lmob[0],
                self.model.gch_params.Ms,
                self.model.gch_params.Ms2,
            )
        )

        if self.model.fl_model.regime == FlowRegime.STATIONARY:
            solve_flow_stationary(
                self.model.grid,
                self.model.fl_model,
                self.model.tr_model,
                unitflw_sources,
                0,
            )
        else:
            # To reproduce HYTEC's behavior -> the initial darcy velocity is null
            grid = self.model.grid
            self.model.fl_model.lu_darcy_x = [np.zeros((grid.nx + 1, grid.ny, grid.nz))]
            self.model.fl_model.lu_darcy_y = [np.zeros((grid.nx, grid.ny + 1, grid.nz))]
            self.model.fl_model.lu_darcy_z = [np.zeros((grid.nx, grid.ny, grid.nz + 1))]
            compute_u_darcy_div(self.model.fl_model, grid, 0)

            # Add the stiffness matrices A and B so that indices match between
            # the forward and the adjoint. This is for development purposes.
            ngc = grid.n_grid_cells

            # only useful for devs or to check the adjoint state correctness
            if self.model.fl_model.is_save_spmats:
                self.model.fl_model.l_q_next.append(sp.sparse.identity(ngc))
                self.model.fl_model.l_q_prev.append(sp.sparse.lil_array((ngc, ngc)))

    def solve(self, is_verbose: bool = False) -> None:
        """
        Solve the forward problem.

        Parameters
        ----------
        is_verbose: bool
            Whether to log info about the coupling convergence (with the
            :mod:`logging` module, at the INFO level). The default is False.

        Raises
        ------
        CouplingConvergenceError
            If the transport-chemistry coupling does not converge, even for the
            smallest timestep allowed.
        """

        self.initialize()
        time_index = 0  # iteration on time

        # Sequential iterative approach with operator splitting
        while self.model.time_params.time_elapsed < self.model.time_params.duration:
            time_index += 1  # Update the number of time iterations
            # Reset numerical acceleration if it was temporarily disabled
            self.model.tr_model.is_num_acc_for_timestep = (
                self.model.tr_model.is_numerical_acceleration
            )
            self._solve_system_for_timestep(time_index, is_verbose)

    def _get_dt_max_cfl(self, time_index: int) -> float:
        """Return the maximum timestep for time ``time_index`` (CFL condition)."""
        if time_index == 1:
            # No velocity field to evaluate the CFL condition from at the first step
            return np.inf
        # The CFL criterion is evaluated based on the previous timestep
        return self.model.time_params.get_dt_max_cfl(self.model, time_index - 1)

    def _solve_system_for_timestep(
        self, time_index: int, is_verbose: bool = False
    ) -> None:
        """
        Solve the flow, the transport and the chemistry for one timestep.

        The timestep is first updated based on the convergence speed of the previous
        one. If the fixed point iterations do not converge, the results of the
        failed attempt are discarded and the timestep is solved again: first without
        numerical acceleration (if it was enabled), and then with a reduced
        timestep.
        """
        time_params = self.model.time_params
        tr_model = self.model.tr_model

        # Do not update the timestep for the first iteration
        # update the timestep based on the convergence speed.
        if time_index != 1:
            time_params.update_dt(
                time_params.nfpi, self._get_dt_max_cfl(time_index), tr_model.max_fpi
            )

        while not self._try_timestep(time_index, is_verbose):
            self._rollback_timestep(time_index)

            if tr_model.is_num_acc_for_timestep:
                # temporary disabling of numerical acceleration
                logger.info(
                    "Timestep %d: no convergence in %d iterations -> disabling the "
                    "numerical acceleration.",
                    time_index,
                    tr_model.max_fpi,
                )
                tr_model.is_num_acc_for_timestep = False
                continue

            if time_params.dt <= time_params.dt_min:
                raise CouplingConvergenceError(
                    f"The transport-chemistry coupling did not converge at time index "
                    f"{time_index} (t = {time_params.time_elapsed:.6g} s) within "
                    f"{tr_model.max_fpi} fixed point iterations for the smallest "
                    f"timestep dt_min = {time_params.dt_min:.6g} s."
                )
            # ask for a decrease of the timestep (n_iter >= max_fpi)
            time_params.update_dt(
                tr_model.max_fpi, self._get_dt_max_cfl(time_index), tr_model.max_fpi
            )
            logger.info(
                "Timestep %d: no convergence -> restarting with dt = %.6g s.",
                time_index,
                time_params.dt,
            )

    def _try_timestep(self, time_index: int, is_verbose: bool = False) -> bool:
        """
        Make one attempt to solve the timestep with the current timestep length.

        Returns
        -------
        bool
            True if the fixed point iterations converged. If False, the model holds
            the results of the failed attempt, which must be discarded with
            :meth:`_rollback_timestep`.
        """
        time_params = self.model.time_params
        fl_model = self.model.fl_model
        tr_model = self.model.tr_model

        # Important: need to save the timestep after the update, otherwise, the
        # wrong timestep is used in the adjoint
        # Save the timesteps to the list of timesteps
        time_params.save_dt()

        # Get the sources
        unitflw_sources_old = fl_model.lunitflow[-1]
        conc_sources_old = tr_model.lsources[-1]
        # Careful, we need to consider the time at the beginning of the timestep
        unitflw_sources, conc_sources = self.model.get_sources(
            np.sum(time_params.ldt[:-1]), self.model.grid
        )

        fl_model.lunitflow.append(unitflw_sources)
        tr_model.lsources.append(conc_sources)

        # Solve the flow -> no iterations with transport/chemistry since we don't have
        # variable permeability nor porosity/diffusion.
        solve_flow_transient_semi_implicit(
            self.model.grid,
            fl_model,
            tr_model,
            unitflw_sources,
            unitflw_sources_old,
            time_params,
            time_index,
        )

        # Now the reactive-transport iterations begin...

        # Reset the number of coupling (Fixed Point) iterations for the current time
        time_params.nfpi = 0

        # Convergence flag -> set to True to skip the chemistry part
        has_converged = tr_model.is_skip_rt

        # Copy the grades (To place in another function afterwards)
        tr_model.limmob.append(tr_model.limmob[time_index - 1].copy())
        tr_model.lmob.append(tr_model.lmob[time_index - 1].copy())

        # Iterate the chemistry transport system while the convergence is no meet
        while not has_converged:
            if time_params.nfpi > tr_model.max_fpi:
                return False

            # Save the grade for the fix point iterations
            tr_model.immob_prev = tr_model.limmob[time_index].copy()

            # One more coupling iteration has been performed
            # Update the number of FPI
            time_params.nfpi += 1

            # Solve the transport
            solve_transport_semi_implicit(
                self.model.grid,
                fl_model,
                tr_model,
                conc_sources,
                conc_sources_old,
                time_params,
                time_index,
                time_params.nfpi,
            )

            # Solve the chemistry
            solve_geochem(
                self.model.grid,
                tr_model,
                self.model.gch_params,
                time_params,
                time_index,
            )

            coupling_error = get_max_coupling_error_forward(tr_model, time_index)
            has_converged = coupling_error < tr_model.fpi_eps
            if is_verbose:
                logger.info(
                    "max-coupling error at it = %d-%d: %s (converged: %s)",
                    time_index,
                    time_params.nfpi,
                    coupling_error,
                    has_converged,
                )

        # Save the number of fixed point iterations required
        time_params.save_nfpi()

        # Update the density for the current timestep
        tr_model.ldensity.append(
            get_density(
                tr_model.lmob[-1],
                self.model.gch_params.Ms,
                self.model.gch_params.Ms2,
            )
        )
        return True

    def _rollback_timestep(self, time_index: int) -> None:
        """
        Discard everything stored by a failed attempt to solve ``time_index``.

        All the lists indexed by time are truncated to the entries preceding
        ``time_index``, so that the timestep can be solved again from a clean state.
        """
        fl_model = self.model.fl_model
        tr_model = self.model.tr_model
        time_params = self.model.time_params

        # Lists with one entry per time (including t=0)
        for lst in (
            fl_model.lhead,
            fl_model.lpressure,
            fl_model.lu_darcy_x,
            fl_model.lu_darcy_y,
            fl_model.lu_darcy_z,
            fl_model.lu_darcy_div,
            fl_model.lunitflow,
            fl_model.l_q_next,
            fl_model.l_q_prev,
            tr_model.lsources,
            tr_model.lmob,
            tr_model.limmob,
        ):
            del lst[time_index:]

        # Lists with one entry per timestep (no entry for t=0)
        for lst in (tr_model.l_q_next, tr_model.l_q_prev, time_params.ldt):
            del lst[time_index - 1 :]

        time_params.nfpi = 0


def get_density(conc: NDArrayFloat, mw1: float, mw2: float) -> NDArrayFloat:
    """
    Compute the density of the solution from the mobile concentrations.

    The density is a linear function of the total dissolved solids (TDS). This is
    implemented only for water solvent.

    Parameters
    ----------
    conc : NDArrayFloat
        Mobile concentrations with shape (2, Nx, Ny, Nz) in mol/kg.
    mw1: float
        Molar weight of the first species (g/mol).
    mw2: float
        Molar weight of the second species (g/mol).

    Returns
    -------
    NDArrayFloat
        Array of densities in kg/m3, with shape (Nx, Ny, Nz).
    """
    # total dissolved solids (kg per kg of water).
    # Add H[+] and OH[-] contributions, which concentrations are not tracked:
    # at pH=7, there are always equal numbers of H+ and OH-, so we take
    # the molar weight of water (sum of the two previous).
    tds = np.full(conc[0].shape, H_PLUS_CONC * WATER_MW)
    # Add species 1 (we divide by 1000 to convert g/mol to kg/mol)
    tds += conc[0] * mw1 / 1000
    # Add species 2
    tds += conc[1] * mw2 / 1000

    # density (linear in the TDS)
    return WATER_DENSITY * (TDS_LINEAR_COEFFICIENT * tds + 1.0)
