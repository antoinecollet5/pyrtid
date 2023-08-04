"""Provide a reactive transport solver."""
from __future__ import annotations

from pyrtid.utils.operators import get_super_lu_preconditioner

from .flow_solver import (
    make_stationary_flow_matrices,
    make_transient_flow_matrices,
    solve_flow_stationary,
    solve_flow_transient_semi_implicit,
)
from .geochem_solver import solve_geochem_explicit
from .models import FlowRegime, ForwardModel
from .transport_solver import (
    make_transport_matrices_diffusion_only,
    solve_transport_semi_implicit,
)


class ForwardSolver:
    """Class solving the reactive transport forward systems."""

    def __init__(self, model: ForwardModel) -> None:
        # The model needs to be copied
        self.model: ForwardModel = model

    def initialize_flow_matrices(self, flow_regime: FlowRegime) -> None:
        """Initialize the matrices to solve the flow problem."""
        if flow_regime == FlowRegime.STATIONARY:
            self.model.fl_model.q_next = make_stationary_flow_matrices(
                self.model.geometry, self.model.fl_model
            )
        if flow_regime == FlowRegime.TRANSIENT:
            (
                self.model.fl_model.q_next,
                self.model.fl_model.q_prev,
            ) = make_transient_flow_matrices(
                self.model.geometry, self.model.fl_model, self.model.time_params
            )

    def initialize_transport_matrices(self) -> None:
        """
        Initialize the trabsport matrices with the diffusion term only.

        The advection term needs to be included at each timestep. Only the diffusion
        part remains constant.
        """
        (
            self.model.tr_model.q_next_diffusion,
            self.model.tr_model.q_prev_diffusion,
        ) = make_transport_matrices_diffusion_only(
            self.model.geometry, self.model.tr_model, self.model.time_params
        )

    def solve(self) -> None:
        """Solve the forward problem."""
        # Reinit all
        self.model.reinit()
        # Update sources
        self.model.fl_model.sources = self.model.get_fl_sources()
        self.model.tr_model.sources = self.model.get_tr_sources()

        # If stationary -> equilibrate the initial heads with sources
        # and boundary conditions
        if self.model.fl_model.regime == FlowRegime.STATIONARY:
            self.initialize_flow_matrices(FlowRegime.STATIONARY)
            solve_flow_stationary(
                self.model.geometry,
                self.model.fl_model,
                get_super_lu_preconditioner(self.model.fl_model.q_next),
                0,
            )

        # Update the flow matrices depending on the flow regime (not modified along
        # the timesteps because permeability and storage coefficients are constant).
        self.initialize_flow_matrices(FlowRegime.TRANSIENT)
        self.initialize_transport_matrices()

        # Preconditioner for the flow matrices (must be done on the fly for transport)
        _flow_q_next_preconditioner = get_super_lu_preconditioner(
            self.model.fl_model.q_next
        )

        # Sequential iterative approach
        # From here the flow is transient
        for time_index in range(1, self.model.time_params.nt + 1):
            # Update the timestep based on the previous iteration
            self.model.time_params.update_dt(1)

            solve_flow_transient_semi_implicit(
                self.model.geometry,
                self.model.fl_model,
                _flow_q_next_preconditioner,
                self.model.time_params,
                time_index,
            )
            solve_transport_semi_implicit(
                self.model.geometry,
                self.model.fl_model,
                self.model.tr_model,
                self.model.time_params,
                time_index,
            )
            solve_geochem_explicit(
                self.model.tr_model,
                self.model.gch_params,
                self.model.time_params,
                time_index,
            )
