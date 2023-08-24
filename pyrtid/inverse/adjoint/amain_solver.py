"""Provide an adjoint solver and model."""
from __future__ import annotations

import logging

import numpy as np

from pyrtid.forward.models import FlowRegime, ForwardModel
from pyrtid.inverse.adjoint.aflow_solver import (
    make_stationary_adj_flow_matrices,
    make_transient_adj_flow_matrices,
    solve_adj_flow_stationary,
    solve_adj_flow_transient_semi_implicit,
    update_adjoint_u_darcy,
)
from pyrtid.inverse.adjoint.ageochem_solver import solve_adj_geochem
from pyrtid.inverse.adjoint.amodels import (
    AdjointFlowModel,
    AdjointModel,
    AdjointTransportModel,
)
from pyrtid.inverse.adjoint.atransport_solver import (
    make_transient_adj_transport_matrices,
    solve_adj_transport_transient_semi_implicit,
)


def get_adjoint_max_coupling_error(
    atr_model: AdjointTransportModel, time_index: int
) -> float:
    r"""
    Return the maximum adjoint chemistry-transport coupling error.

    The fixed point iteration convergence criteria reads:

    .. math::
        \text{max} \left\lVert 1 - \dfrac{\lambda_{c}^{n, k+1}}
        {\lambda_{c}^{n, k}} \right\rVert  < \epsilon

    with $k$ the number of fixed point iterations.

    This error is evaluated from the mobile adjoint concentrations.
    """
    return float(
        np.nan_to_num(
            np.max(
                np.abs(1 - atr_model.a_conc[:, :, time_index] / atr_model.a_conc_prev)
            ),
            nan=0.0,
        )
    )


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
                self.adj_model.a_fl_model.q_next,
                self.adj_model.a_fl_model.q_prev,
            ) = make_stationary_adj_flow_matrices(
                self.fwd_model.geometry,
                self.fwd_model.fl_model,
                self.fwd_model.time_params,
            )
        if flow_regime == FlowRegime.TRANSIENT:
            (
                self.adj_model.a_fl_model.q_next,
                self.adj_model.a_fl_model.q_prev,
            ) = make_transient_adj_flow_matrices(
                self.fwd_model.geometry,
                self.fwd_model.fl_model,
                self.fwd_model.time_params,
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

    def solve(self, is_verbose: bool = False) -> None:
        """
        Solve the adjoint system of equations.
        """

        # Initiate adjoint concentrations and grades
        # _init_adjoint_variables(
        #     _tr_model,
        #     a_tr_model,
        #     _gch_params,
        #     geometry,
        #     _a_sources,
        #     _time_params,
        #     "direct",
        #     True,
        # )

        # Construct the flow matrices (not modified along the timesteps because
        # permeability and storage coefficients are constant).
        self.initialize_ajd_flow_matrices(FlowRegime.TRANSIENT)

        # Initialize transport matrices with diffusion (advection is added on the fly)
        # Consequently, the preconditioner is built on the fly too.
        self.initialize_ajd_transport_matrices()

        for time_index in range(
            self.fwd_model.time_params.nts - 1, 0, -1
        ):  # Reverse order in time, and reverse order in operator sequence
            self._solve_system_for_timestep(time_index, is_verbose)

        # Flow: solve for the last timestep, only if the flow was initially stationnary
        # Otherwise, just copy as for transport
        _copy_fl_adj_prev_to_current(self.adj_model.a_fl_model, 0)
        if self.fwd_model.fl_model.regime == FlowRegime.STATIONARY:
            self.initialize_ajd_flow_matrices(FlowRegime.STATIONARY)
            solve_adj_flow_stationary(
                self.fwd_model.geometry,
                self.fwd_model.fl_model,
                self.adj_model.a_fl_model,
                0,  # time index
            )

    def _solve_system_for_timestep(
        self, time_index: int, is_verbose: bool = False
    ) -> None:
        # Some references.
        a_tr_model = self.adj_model.a_tr_model

        nafpi = 1  # number of coupling (Fixed Point) iterations
        # Convergence flag for the adjoint

        # Copy the grades (To place in another function afterwards)
        a_tr_model.a_conc[:, :, time_index] = a_tr_model.a_conc[:, :, time_index + 1]

        has_converged = False

        # print(atr_model.a_conc[:, :, time_index])

        # Iterate the chemistry transport system while the convergence is no meet
        while not has_converged:
            # Save the grade for the fix point iterations
            a_tr_model.a_conc_prev = a_tr_model.a_conc[:, :, time_index].copy()

            # 1) Start by solving adjoint geochemistry
            solve_adj_geochem(
                self.fwd_model.tr_model,
                self.adj_model.a_tr_model,
                self.fwd_model.gch_params,
                self.fwd_model.geometry,
                self.fwd_model.time_params,
                time_index,
            )

            # solve_adj_transport_transient_semi_implicit(
            #     geometry, _tr_model, a_tr_model, _a_sources, time_params, time_index
            # )

            # 2) Solve the adjoint transport
            solve_adj_transport_transient_semi_implicit(
                self.fwd_model.geometry,
                self.fwd_model.fl_model,
                self.fwd_model.tr_model,
                self.adj_model.a_tr_model,
                self.fwd_model.time_params,
                time_index,
            )

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
            solve_adj_flow_transient_semi_implicit(
                self.fwd_model.geometry,
                self.fwd_model.fl_model,
                self.adj_model.a_fl_model,
                self.fwd_model.time_params,
                time_index,
                self.adj_model.is_mob_obs,
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
