"""Provide an adjoint solver and model."""
from __future__ import annotations

from pyrtid.forward.models import FlowRegime, ForwardModel
from pyrtid.inverse.adjoint.aflow_solver import (
    make_stationary_adj_flow_matrices,
    make_transient_adj_flow_matrices,
    solve_adj_flow_stationary,
    solve_adj_flow_transient_semi_implicit,
    update_adjoint_u_darcy,
)
from pyrtid.inverse.adjoint.ageochem_solver import solve_adj_geochem_explicit
from pyrtid.inverse.adjoint.amodels import (
    AdjointFlowModel,
    AdjointModel,
    AdjointTransportModel,
)
from pyrtid.inverse.adjoint.atransport_solver import (
    make_transient_adj_transport_matrices,
    solve_adj_transport_transient_semi_implicit,
)
from pyrtid.utils.operators import get_super_lu_preconditioner


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
            _description_
        adj_model : AdjointModel
            _description_
        """
        self.fwd_model: ForwardModel = fwd_model
        self.adj_model: AdjointModel = adj_model

    def initialize_ajd_flow_matrices(self, flow_regime: FlowRegime) -> None:
        """Initialize matrices to solve the adjoint flow problem."""
        if flow_regime == FlowRegime.STATIONARY:
            (
                self.adj_model.fl_model.q_next,
                self.adj_model.fl_model.q_prev,
            ) = make_stationary_adj_flow_matrices(
                self.fwd_model.geometry,
                self.fwd_model.fl_model,
                self.fwd_model.time_params,
            )
        if flow_regime == FlowRegime.TRANSIENT:
            (
                self.adj_model.fl_model.q_next,
                self.adj_model.fl_model.q_prev,
            ) = make_transient_adj_flow_matrices(
                self.fwd_model.geometry,
                self.fwd_model.fl_model,
                self.fwd_model.time_params,
            )

    def initialize_ajd_transport_matrices(self) -> None:
        """Initialize matrices to solve the adjoint transport problem."""
        (
            self.adj_model.tr_model.q_next_diffusion,
            self.adj_model.tr_model.q_prev_diffusion,
        ) = make_transient_adj_transport_matrices(
            self.fwd_model.geometry,
            self.fwd_model.tr_model,
            self.fwd_model.time_params,
        )

    def solve(self) -> None:
        """
        Solve the adjoint diffusion equation.

        daT/dt = - div D grad (aT) - dT
        with aT(:,t=T) = 0 and daT/dn = 0 on the edges

        """
        # Construct the flow matrices (not modified along the timesteps because
        # permeability and storage coefficients are constant).
        self.initialize_ajd_flow_matrices(FlowRegime.TRANSIENT)
        _flow_q_next_preconditioner = get_super_lu_preconditioner(
            self.adj_model.fl_model.q_next
        )

        # Initialize transport matrices with diffusion (advection is added on the fly)
        # Consequently, the preconditioner is built on the fly too.
        self.initialize_ajd_transport_matrices()

        # TODO: for now there is no retroaction from the chemistry to the flow

        for time_index in range(
            self.fwd_model.time_params.nt, 0, -1
        ):  # Reverse order in time, and reverse order in operator sequence
            # 0) copy last transport state
            _copy_tr_adj_prev_to_current(self.adj_model.tr_model, time_index)

            # 1) Solve the transport
            # TODO: Add a "if mob obs" ??
            solve_adj_transport_transient_semi_implicit(
                self.fwd_model.geometry,
                self.fwd_model.fl_model,
                self.fwd_model.tr_model,
                self.adj_model.tr_model,
                self.fwd_model.time_params.dt,
                time_index,
            )

            # 2) Solve the chemistry
            solve_adj_geochem_explicit(
                self.fwd_model.tr_model,
                self.adj_model.tr_model,
                self.fwd_model.gch_params,
                self.fwd_model.geometry,
                self.fwd_model.time_params,
                time_index,
            )

            # Need to compute the adjoint darcy velocities
            update_adjoint_u_darcy(
                self.fwd_model.geometry,
                self.fwd_model.tr_model,
                self.adj_model.tr_model,
                self.fwd_model.fl_model,
                self.adj_model.fl_model,
                time_index,
            )

            # 3) Solve the flow last -> requires the previous
            solve_adj_flow_transient_semi_implicit(
                self.fwd_model.geometry,
                self.fwd_model.fl_model,
                self.adj_model.fl_model,
                _flow_q_next_preconditioner,
                time_index,
                self.adj_model.is_mob_obs,
            )

        # Transport: do not solve for the last timestep, simply copy the results
        # This is the right way to get consistent with the mineral gradient
        _copy_tr_adj_prev_to_current(self.adj_model.tr_model, 0)

        # Flow: solve for the last timestep, only if the flow was initially stationnary
        # Otherwise, just copy as for transport
        _copy_fl_adj_prev_to_current(self.adj_model.fl_model, 0)
        if self.fwd_model.fl_model.regime == FlowRegime.STATIONARY:
            self.initialize_ajd_flow_matrices(FlowRegime.STATIONARY)
            solve_adj_flow_stationary(
                self.fwd_model.geometry,
                self.fwd_model.fl_model,
                self.adj_model.fl_model,
                _flow_q_next_preconditioner,
                0,  # time index
            )


def _copy_tr_adj_prev_to_current(
    a_tr_model: AdjointTransportModel, time_index: int
) -> None:
    try:
        # Copy the last index
        a_tr_model.a_conc[:, :, time_index] = a_tr_model.a_conc[:, :, time_index + 1]
        a_tr_model.a_grade[:, :, time_index] = a_tr_model.a_grade[:, :, time_index + 1]
    except IndexError:
        # Do nothing for the first timestep (keep 0)
        pass


def _copy_fl_adj_prev_to_current(a_fl_model: AdjointFlowModel, time_index: int) -> None:
    try:
        # Copy the last index
        a_fl_model.a_head[:, :, time_index] = a_fl_model.a_head[:, :, time_index + 1]
        a_fl_model.a_u_darcy_x[:, :, time_index] = a_fl_model.a_u_darcy_x[
            :, :, time_index + 1
        ]
        a_fl_model.a_u_darcy_y[:, :, time_index] = a_fl_model.a_u_darcy_y[
            :, :, time_index + 1
        ]
    except IndexError:
        # Do nothing for the first timestep (keep 0)
        pass
