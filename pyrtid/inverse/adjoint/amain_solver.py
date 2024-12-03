"""Provide an adjoint solver and model."""

from __future__ import annotations

import logging
from typing import Optional

from pyrtid.forward.models import ForwardModel
from pyrtid.inverse.adjoint.adensity_solver import solve_adj_density
from pyrtid.inverse.adjoint.aflow_solver import (
    make_initial_adj_flow_matrices,
    make_transient_adj_flow_matrices,
    solve_adj_flow,
    update_adjoint_u_darcy,
)
from pyrtid.inverse.adjoint.ageochem_solver import solve_adj_geochem
from pyrtid.inverse.adjoint.amodels import AdjointModel
from pyrtid.inverse.adjoint.atransport_solver import (
    get_adjoint_max_coupling_error,
    make_transient_adj_transport_matrices,
    solve_adj_transport_transient_semi_implicit,
)
from pyrtid.inverse.obs import Observables


class AdjointSolver:
    """Solve the adjoint reactive-transport problem."""

    __slots__ = ["fwd_model", "adj_model"]

    def __init__(
        self,
        fwd_model: ForwardModel,
        adj_model: AdjointModel,
    ) -> None:
        """
        Initialize the instance.

        Parameters
        ----------
        fwd_model : ForwardModel
            The forward model instance.
        adj_model : AdjointModel
            The adjoint model instance.
        """
        self.fwd_model: ForwardModel = fwd_model
        self.adj_model: AdjointModel = adj_model

        if (
            self.adj_model.a_fl_model.is_use_continuous_adj
            and not self.fwd_model.tr_model.is_skip_rt
        ):
            raise ValueError(
                "Continuous adjoint only working if reactive transport is skipped!"
            )

    def initialize_ajd_flow_matrices(self) -> None:
        """Initialize matrices to solve the adjoint flow problem."""
        (
            self.adj_model.a_fl_model.q_next_init,
            self.adj_model.a_fl_model.q_prev_init,
        ) = make_initial_adj_flow_matrices(
            self.fwd_model.geometry,
            self.fwd_model.fl_model,
            self.fwd_model.tr_model,
            self.adj_model.a_fl_model,
            self.fwd_model.time_params,
            is_q_prev_for_gradient=False,
        )
        (
            self.adj_model.a_fl_model.q_next,
            self.adj_model.a_fl_model.q_prev,
        ) = make_transient_adj_flow_matrices(
            self.fwd_model.geometry,
            self.fwd_model.fl_model,
            self.fwd_model.tr_model,
            self.adj_model.a_fl_model,
            self.fwd_model.time_params,
            self.fwd_model.time_params.nts,
        )

    def initialize_ajd_transport_matrices(self) -> None:
        """Initialize matrices to solve the adjoint transport problem."""
        (
            self.adj_model.a_tr_model.q_next_diffusion,
            self.adj_model.a_tr_model.q_prev_diffusion,
        ) = make_transient_adj_transport_matrices(
            self.fwd_model.geometry,
            self.fwd_model.tr_model,
            self.fwd_model.time_params,
        )

    def solve(
        self,
        observables: Observables,
        hm_end_time: Optional[float] = None,
        is_verbose: bool = False,
        max_nafpi: int = 30,
    ) -> None:
        """
        Solve the adjoint system of equations.

        Parameters
        ----------
        is_verbose : bool, optional
            Whether to display computation infrmation, by default False.
        """
        # Reset all adjoint variables to zero.
        self.adj_model.reinit()

        # Compute the adjoint sources
        self.adj_model.init_adjoint_sources(
            self.fwd_model,
            observables,
            hm_end_time=hm_end_time,
        )

        # Construct the flow matrices (not modified along the timesteps because
        # permeability and storage coefficients are constant).
        self.initialize_ajd_flow_matrices()

        # Initialize transport matrices with diffusion (advection is added on the fly)
        # Consequently, the preconditioner is built on the fly too.
        self.initialize_ajd_transport_matrices()

        for time_index in range(
            self.fwd_model.time_params.nts,
            -1,
            -1,  # type: ignore
        ):  # Reverse order in time, and reverse order in operator sequence
            self.adj_model.a_tr_model.is_adj_num_acc_for_timestep = (
                self.adj_model.a_tr_model.is_adj_numerical_acceleration
            )
            self._solve_system_for_timestep(time_index, is_verbose, max_nafpi)

    def _solve_system_for_timestep(
        self, time_index: int, is_verbose: bool = False, max_nafpi: int = 30
    ) -> None:
        # Some references.
        a_tr_model = self.adj_model.a_tr_model
        nafpi = 1  # number of coupling (Fixed Point) iterations
        # set to true to skip the chemistry
        has_converged = self.fwd_model.tr_model.is_skip_rt

        # 1) Start by computing the adjoint density which is the last thing computed
        # in the forward problem.
        solve_adj_density(
            self.fwd_model.fl_model,
            self.fwd_model.tr_model,
            self.adj_model.a_fl_model,
            self.adj_model.a_tr_model,
            time_index,
            self.fwd_model.time_params,
            self.fwd_model.geometry,
            self.fwd_model.gch_params.Ms,
        )

        # 2) Iterate the chemistry transport system while the convergence is no meet
        while not has_converged:
            if nafpi > max_nafpi:
                if self.adj_model.a_tr_model.is_adj_num_acc_for_timestep:
                    # temporary disabling of numerical acceleration
                    self.adj_model.a_tr_model.is_adj_num_acc_for_timestep = False
                    nafpi = 1
                # restart the timestep
                else:
                    raise RuntimeError(
                        f"The adjoint fixed point loop at time iteration {time_index}"
                        f" (t={self.fwd_model.time_params.times[time_index]}), exceeded"
                        f"the maximum number of fpi iterations allowed ({max_nafpi}!)\n"
                        f"The convergence criteria might be too low. Try to diminish "
                        "it or increase the maximum number of fpi iterations!"
                    )

            # 2.1) Start by solving adjoint geochemistry
            solve_adj_geochem(
                self.fwd_model.tr_model,
                self.adj_model.a_tr_model,
                self.fwd_model.gch_params,
                self.fwd_model.geometry,
                self.fwd_model.time_params,
                time_index,
                nafpi=nafpi,
            )

            # 2.2) Solve the adjoint transport
            solve_adj_transport_transient_semi_implicit(
                self.fwd_model.geometry,
                self.fwd_model.fl_model,
                self.fwd_model.tr_model,
                self.adj_model.a_tr_model,
                self.fwd_model.time_params,
                time_index,
                self.fwd_model.gch_params,
                nafpi=nafpi,
            )

            # One more coupling iteration has been performed
            if is_verbose:
                logging.info(
                    f"max-coupling error at time_index = {time_index}-{nafpi}: "
                    f"{get_adjoint_max_coupling_error(a_tr_model, time_index)}"
                )
            has_converged = (
                get_adjoint_max_coupling_error(a_tr_model, time_index)
                < self.adj_model.a_tr_model.afpi_eps
            )
            if is_verbose:
                logging.info(f"has-converged ?: {has_converged}")

            # Update the number of FPI
            nafpi += 1

        # 3) Need to compute the adjoint darcy velocities
        update_adjoint_u_darcy(
            self.fwd_model.geometry,
            self.fwd_model.tr_model,
            self.adj_model.a_tr_model,
            self.fwd_model.fl_model,
            self.adj_model.a_fl_model,
            time_index,
        )

        # 4) Solve the flow last -> requires the previous
        solve_adj_flow(
            self.fwd_model.geometry,
            self.fwd_model.fl_model,
            self.fwd_model.tr_model,
            self.adj_model.a_fl_model,
            self.fwd_model.time_params,
            time_index,
        )
