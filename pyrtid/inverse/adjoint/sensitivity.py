"""Provide a reactive transport solver."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from pyrtid.forward.flow_solver import solve_fl_gmres
from pyrtid.forward.models import FlowRegime, ForwardModel
from pyrtid.forward.solver import ForwardSolver
from pyrtid.inverse.obs import (
    Observables,
    get_observables_values_as_1d_vector,
    get_predictions_matching_observations,
)
from pyrtid.utils import NDArrayFloat, dxi_harmonic_mean


class ForwardSensitivitySolver:
    """Class solving the reactive transport forward systems."""

    def __init__(self, model: ForwardModel) -> None:
        # The model needs to be copied
        self.model: ForwardModel = model
        self.solver = ForwardSolver(model)

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
        # This is required to avoid recomputing the preconditioner each time
        self.model.fl_model.is_save_spilu = True

        # solve for t=0
        self.solver.initialize()
        time_index = 0  # iteration on time

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

        # TODO: initialize z_tmp with all forward variables
        # -> double the size on the ram
        # There are as many z as there are models.
        z_tmp = np.zeros(
            (self.model.geometry.nx, self.model.geometry.ny, ne), dtype=np.float64
        )

        # compute the sensitivities for t=0
        # z is the working vector
        z_tmp = self.solve_sensitivities(vecs, z_tmp, time_index)

        # Fill jacvecs: NDArrayFloat

        # Sequential iterative approach with operator splitting
        while self.model.time_params.time_elapsed < self.model.time_params.duration:
            time_index += 1  # Update the number of time iterations
            # Reset numerical acceleration if it was temporarily disabled
            self.model.tr_model.is_num_acc_for_timestep = (
                self.model.tr_model.is_numerical_acceleration
            )
            self.solver._solve_system_for_timestep(time_index, is_verbose)

            z_tmp = self.solve_sensitivities(vecs, z_tmp, time_index)

        # get the predictions
        d_pred = get_predictions_matching_observations(
            self.model, observables, hm_end_time
        )

        return d_pred, jacvecs

    def solve_sensitivities(
        self, vecs: NDArrayFloat, z_tmp: NDArrayFloat, time_index: int
    ) -> NDArrayFloat:
        # step 1: compte - (d F^n(s, u) / d s) @ v
        # note that this is the same as computing the derivative of
        # d < \lambda_u, F^n(s, u) > / ds
        # So we can reuse the code for the adjoint state, replacing the adjoint variable
        # with the vector we are interested in.
        rhs = -dFhdKv(self.model, time_index, vecs)

        # add -B^{n-1} z^{n-1}
        rhs -= self.model.fl_model.q_prev @ z_tmp

        # Solve A^n z^n = rhs
        z_tmp, exit_code = solve_fl_gmres(
            self.model.fl_model,
            rhs,
            self.model.fl_model.super_ilu,
            self.model.fl_model.preconditioner,
        )

        return z_tmp

    def fill_jacvecs(
        self, jacvecs: NDArrayFloat, z_tmp: NDArrayFloat, time_index: int
    ) -> None:
        # Derivative of the observations with respect to h^{n} @ z_tmp
        # For now do nothing
        jacvecs[:, :] = jacvecs[:, :]


def dFhdKv(
    fwd_model: ForwardModel, time_index: int, vecs: NDArrayFloat
) -> NDArrayFloat:
    """
    Compute the gradient with respect to the permeability using head observations.

    Parameters
    ----------
    fwd_model : ForwardModel
        The forward model which contains all forward variables and parameters.
    adj_model : AdjointModel
        The adjoint model which contains all adjoint variables and parameters.

    Returns
    -------
    NDArrayFloat
        Gradient with respect to the permeability using head observations.
    """
    shape = (fwd_model.geometry.nx, fwd_model.geometry.ny)
    permeability = fwd_model.fl_model.permeability
    crank_flow: float = fwd_model.fl_model.crank_nicolson

    head = fwd_model.fl_model.lhead[time_index]
    if time_index != 0:
        head_prev = fwd_model.fl_model.lhead[time_index - 1]
    else:
        # make a reference just for linting
        # it won't be used anyway
        head_prev = head

    grad = np.zeros(shape)

    # Consider the x axis
    if shape[0] > 1:
        dKijdKxv = (
            dxi_harmonic_mean(permeability[1:, :], permeability[:-1, :]) * vecs[1:, :]
            + dxi_harmonic_mean(permeability[:-1, :], permeability[1:, :])
            * vecs[:-1, :]
        )
        tmp = fwd_model.geometry.gamma_ij_x / fwd_model.geometry.dx

        # For all n != 0
        if time_index != 0:
            lhs = (
                (
                    crank_flow * (head[1:, :] - head[:-1, :])
                    + (1.0 - crank_flow) * (head_prev[1:, :] - head_prev[:-1, :])
                )
                * tmp
                / fwd_model.geometry.grid_cell_volume
            )
            # Forward
            grad[:-1, :] -= (
                lhs / fwd_model.fl_model.storage_coefficient[:-1, :] * dKijdKxv
            )
            # Backward scheme
            grad[1:, :] += (
                lhs / fwd_model.fl_model.storage_coefficient[1:, :] * dKijdKxv
            )

        # Handle the stationary case for n == 0
        elif fwd_model.fl_model.regime == FlowRegime.STATIONARY:
            # Forward
            lhs = (
                (head[1:, :] - head[:-1, :])
                * tmp
                / fwd_model.geometry.grid_cell_volume
                * dKijdKxv
            )
            grad[:-1, :] -= lhs
            grad[1:, :] += lhs

    # Consider the y axis for 2D cases
    # Consider the x axis
    if shape[1] > 1:
        dKijdKyv = (
            dxi_harmonic_mean(permeability[:, 1:], permeability[:, :-1]) * vecs[:, 1:]
            + dxi_harmonic_mean(permeability[:, :-1], permeability[:, 1:])
            * vecs[:, :-1]
        )
        tmp = fwd_model.geometry.gamma_ij_y / fwd_model.geometry.dy

        # For all n != 0
        if time_index != 0:
            lhs = (
                (
                    crank_flow * (head[:, 1:] - head[:, :-1])
                    + (1.0 - crank_flow) * (head_prev[:, 1:] - head_prev[:, :-1])
                )
                * tmp
                / fwd_model.geometry.grid_cell_volume
            )
            # Forward
            grad[:, :-1] -= (
                lhs / fwd_model.fl_model.storage_coefficient[:, :-1] * dKijdKyv
            )
            # Backward scheme
            grad[:, 1:] += (
                lhs / fwd_model.fl_model.storage_coefficient[:, 1:] * dKijdKyv
            )

        # Handle the stationary case for n == 0
        elif fwd_model.fl_model.regime == FlowRegime.STATIONARY:
            # Forward
            lhs = (
                (head[:, 1:] - head[:, :-1])
                * tmp
                / fwd_model.geometry.grid_cell_volume
                * dKijdKyv
            )
            grad[:, :-1] -= lhs
            grad[:, 1:] += lhs

    grad[
        fwd_model.fl_model.cst_head_indices[0], fwd_model.fl_model.cst_head_indices[1]
    ] = 0

    return grad
