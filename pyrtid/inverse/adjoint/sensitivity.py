"""Provide a reactive transport solver."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

from pyrtid.forward.flow_solver import (
    compute_u_darcy_div,
    make_stationary_flow_matrices,
    make_transient_flow_matrices,
    solve_flow_stationary,
    solve_flow_transient_semi_implicit,
)
from pyrtid.forward.geochem_solver import solve_geochem
from pyrtid.forward.models import FlowRegime, ForwardModel
from pyrtid.forward.solver import (
    ForwardSolver,
    get_density,
    get_max_coupling_error_forward,
)
from pyrtid.forward.transport_solver import solve_transport_semi_implicit
from pyrtid.inverse.obs import (
    Observables,
    get_observables_values_as_1d_vector,
    get_predictions_matching_observations,
)
from pyrtid.utils import NDArrayFloat


class ForwardSensitivitySolver:
    """Class solving the reactive transport forward systems."""

    def __init__(self, model: ForwardModel) -> None:
        # The model needs to be copied
        self.model: ForwardModel = model
        self.solver = ForwardSolver(model)

    def initialize_flow_matrices(self, flow_regime: FlowRegime) -> None:
        """Initialize the matrices to solve the flow problem."""
        if flow_regime == FlowRegime.STATIONARY:
            self.model.fl_model.q_next = make_stationary_flow_matrices(
                self.model.geometry, self.model.fl_model
            )
        # no need to initialize the flow matrices if gravity is on because since the
        # density varies with time, the matrix is recreated at each timestep.
        if flow_regime == FlowRegime.TRANSIENT and not self.model.fl_model.is_gravity:
            (
                self.model.fl_model.q_next,
                self.model.fl_model.q_prev,
            ) = make_transient_flow_matrices(
                self.model.geometry,
                self.model.fl_model,
                self.model.tr_model,
                self.model.time_params.nts,
            )

    def solve(
        self,
        observables: Observables,
        vecs: NDArrayFloat,
        hm_end_time: Optional[float] = None,
        is_verbose: bool = False,
    ) -> Tuple[NDArrayFloat, NDArrayFloat]:
        """
        Solve the forward problem and apply the forward sensitivity method.

        Parameters
        ----------
        observables:

        vecs: NDArrayFloat
            Vectors to multiply with the Jacobian matrix, i.e., derivatives of the
            simulated values matching the observations with respect to the parameters
            of interest.

        hm_end_time:
            The default is None.

        is_verbose: bool
            Whether to display info. The default is False.

        Returns
        -------
        The products between the Jacobian matrix of the observations and
        """
        # Reinit all
        self.model.reinit()

        # number of observations
        n_obs = get_observables_values_as_1d_vector(observables, hm_end_time).size
        # number of adjusted values and number of rhs vectors to be multiplied with the
        # Jacobian matrix.
        ns, ne = vecs.shape

        # Array in which the Ne products with the Jacobian matrix are stored.
        # This matrix has shape (N_obs, Ne)
        jacvecs: NDArrayFloat = np.zeros(
            (n_obs, ne),
        )

        # If stationary -> equilibrate the initial heads with sources
        # and boundary conditions

        # TODO: on the fly
        # # Compute the adjoint sources
        # self.adj_model.init_adjoint_sources(
        #     self.fwd_model,
        #     observables,
        #     hm_end_time=hm_end_time,
        # )

        # Get the flow and concentration sources
        unitflw_sources, conc_sources = self.model.get_sources(
            self.model.time_params.time_elapsed, self.model.geometry
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
            self.initialize_flow_matrices(FlowRegime.STATIONARY)
            solve_flow_stationary(
                self.model.geometry,
                self.model.fl_model,
                self.model.tr_model,
                unitflw_sources,
                0,
            )
        else:
            # To reproduce HYTEC's behavior -> the initial darcy velocity is null
            self.model.fl_model.lu_darcy_x = [
                np.zeros((self.model.geometry.nx + 1, self.model.geometry.ny))
            ]
            self.model.fl_model.lu_darcy_y = [
                np.zeros((self.model.geometry.nx, self.model.geometry.ny + 1))
            ]
            compute_u_darcy_div(self.model.fl_model, self.model.geometry, 0)

        # Update the flow matrices depending on the flow regime (not modified along
        # the timesteps because permeability and storage coefficients are constant).
        self.initialize_flow_matrices(FlowRegime.TRANSIENT)

        time_index = 0  # iteration on time

        # Sequential iterative approach with operator splitting
        while self.model.time_params.time_elapsed < self.model.time_params.duration:
            time_index += 1  # Update the number of time iterations
            # Reset numerical acceleration if it was temporarily disabled
            self.model.tr_model.is_num_acc_for_timestep = (
                self.model.tr_model.is_numerical_acceleration
            )
            self._solve_system_for_timestep(time_index, is_verbose)

        # get the predictions
        d_pred = get_predictions_matching_observations(
            self.model, observables, hm_end_time
        )

        return d_pred, jacvecs

    def _solve_system_for_timestep(
        self, time_index: int, is_verbose: bool = False
    ) -> None:
        # Do not update the timestep for the first iteration
        # update the timestep based on the convergence speed.
        if time_index != 1:
            # The CFL criterion is evaluated based on the previous timestep
            dt_max_cfl = self.model.time_params.get_dt_max_cfl(
                self.model, time_index - 1
            )
            self.model.time_params.update_dt(
                self.model.time_params.nfpi, dt_max_cfl, self.model.tr_model.max_fpi
            )
        # Important: need to save the timestep after the update, otherwise, the
        # wrong timestep is used in the adjoint
        # Save the timesteps to the list of timesteps
        self.model.time_params.save_dt()

        # Get the sources
        unitflw_sources_old = self.model.fl_model.lunitflow[-1]
        conc_sources_old = self.model.tr_model.lsources[-1]
        unitflw_sources, conc_sources = self.model.get_sources(
            self.model.time_params.time_elapsed, self.model.geometry
        )

        self.model.fl_model.lunitflow.append(unitflw_sources)
        self.model.tr_model.lsources.append(conc_sources)

        # Solve the flow -> no iterations with transport/chemistry since we don't have
        # variable permeability nor porosity/diffusion.
        solve_flow_transient_semi_implicit(
            self.model.geometry,
            self.model.fl_model,
            self.model.tr_model,
            unitflw_sources,
            unitflw_sources_old,
            self.model.time_params,
            time_index,
        )

        # Now the reactive-transport iterations begin...

        # Reset the number of coupling (Fixed Point) iterations for the current time
        self.model.time_params.nfpi = 0

        # Convergence flag -> set to True to skip the chemistry part
        has_converged = self.model.tr_model.is_skip_rt

        # Copy the grades (To place in another function afterwards)
        self.model.tr_model.limmob.append(
            self.model.tr_model.limmob[time_index - 1].copy()
        )
        self.model.tr_model.lmob.append(self.model.tr_model.lmob[time_index - 1].copy())

        # Iterate the chemistry transport system while the convergence is no meet
        while not has_converged:
            if self.model.time_params.nfpi > self.model.tr_model.max_fpi:
                if self.model.tr_model.is_num_acc_for_timestep:
                    # temporary disabling of numerical acceleration
                    self.model.tr_model.is_num_acc_for_timestep = False
                # restart the timestep
                self._solve_system_for_timestep(time_index, is_verbose)

            # Save the grade for the fix point iterations
            self.model.tr_model.immob_prev = self.model.tr_model.limmob[
                time_index
            ].copy()

            # One more coupling iteration has been performed
            # Update the number of FPI
            self.model.time_params.nfpi += 1

            # Solve the transport
            solve_transport_semi_implicit(
                self.model.geometry,
                self.model.fl_model,
                self.model.tr_model,
                conc_sources,
                conc_sources_old,
                self.model.time_params,
                time_index,
                self.model.time_params.nfpi,
            )

            # Solve the chemistry
            solve_geochem(
                self.model.tr_model,
                self.model.gch_params,
                self.model.time_params,
                time_index,
            )

            if is_verbose:
                logging.info(
                    f"max-coupling error at it = {time_index}"
                    f"-{self.model.time_params.nfpi}:"
                    f"{get_max_coupling_error_forward(self.model.tr_model, time_index)}"
                )
            has_converged = (
                get_max_coupling_error_forward(self.model.tr_model, time_index)
                < self.model.tr_model.fpi_eps
            )
            if is_verbose:
                logging.info(f"has-converged ?: {has_converged}")

        # Save the number of fixed point iterations required
        self.model.time_params.save_nfpi()

        # Update the density for the current timestep
        self.model.tr_model.ldensity.append(
            get_density(
                self.model.tr_model.lmob[-1],
                self.model.gch_params.Ms,
                self.model.gch_params.Ms2,
            )
        )
