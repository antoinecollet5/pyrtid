"""Provide a reactive transport solver."""

from __future__ import annotations

import logging

import numpy as np

from pyrtid.utils import NDArrayFloat

from .flow_solver import (
    compute_u_darcy_div,
    make_stationary_flow_matrices,
    make_transient_flow_matrices,
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
from .transport_solver import (
    make_transport_matrices_diffusion_only,
    solve_transport_semi_implicit,
)


def get_max_coupling_error(current_arr, prev_arr) -> float:
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

    def solve(self, is_verbose: bool = False) -> None:
        """Solve the forward problem.

        Parameters
        ----------
        is_verbose: bool
            Whether to display info. The default is False.
        """
        # Reinit all
        self.model.reinit()

        # If stationary -> equilibrate the initial heads with sources
        # and boundary conditions

        # Get the flow and concentration sources
        flw_sources, conc_sources = self.model.get_sources(
            self.model.time_params.time_elapsed, self.model.geometry
        )
        self.model.fl_model.lunitflow.append(flw_sources)
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
                flw_sources,
                0,
            )
        else:
            # To reproduce HYTEC's behavior -> the initial darcy velocity is null
            # compute_u_darcy(
            #     self.model.fl_model, self.model.tr_model, self.model.geometry, 0
            # )
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
        self.initialize_transport_matrices()

        time_index = 0  # iteration on time

        # Sequential iterative approach with operator splitting
        while self.model.time_params.time_elapsed < self.model.time_params.duration:
            time_index += 1  # Update the number of time iterations
            # Reset numerical acceleration if it was temporarily disabled
            self.model.tr_model.is_num_acc_for_timestep = (
                self.model.tr_model.is_numerical_acceleration
            )
            self._solve_system_for_timestep(time_index, is_verbose)

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
        flw_sources_old = self.model.fl_model.lunitflow[-1]
        conc_sources_old = self.model.tr_model.lsources[-1]
        flw_sources, conc_sources = self.model.get_sources(
            self.model.time_params.time_elapsed, self.model.geometry
        )

        self.model.fl_model.lunitflow.append(flw_sources)
        self.model.tr_model.lsources.append(conc_sources)

        # Solve the flow -> no iterations with transport/chemistry since we don't have
        # variable permeability nor porosity/diffusion.
        solve_flow_transient_semi_implicit(
            self.model.geometry,
            self.model.fl_model,
            self.model.tr_model,
            flw_sources,
            flw_sources_old,
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


def get_density(conc: NDArrayFloat, mw1: float, mw2: float) -> NDArrayFloat:
    """

    Parameters
    ----------
    conc : NDArrayFloat
        Array of concentration (3D) with shape (2, Nx, Ny) in mol/kg.
    mv1: float
        Molar weight of the first species (g/mol).
    mv2: float
        Molar weight of the second species (g/mol).

    Returns
    -------
    NDArrayFloat
        2D array of densities with shape (Nx, Ny).
    """

    # total dissolved fraction
    # Add H[+] and OH[-] contributions which concentrations are not
    # At pH=7, there are always equal numbers of H + and OH -, so we take
    # the molar weight of water (sum of the two previous).
    # tds += H_PLUS_CONC * WATER_MW;
    tds = np.ones(conc[0].shape) * H_PLUS_CONC * WATER_MW
    # Add species 1
    tds += conc[0] * mw1 / 1000  # we divide per 1000 because we need kg/mol
    # Add species 2
    tds += conc[1] * mw2 / 1000  # we divide per 1000 because we need kg/mol

    # 2) compute the density (this is implemented only for water solvent)
    # in kg/l
    # tds[:, :] = (
    #     0.1 * time_index # * np.random.default_rng(time_index).random()
    # )  # * np.random.default_rng(time_index).random(size=tds.shape)
    return WATER_DENSITY * (TDS_LINEAR_COEFFICIENT * tds + 1.0)
