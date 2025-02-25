"""Provide a forward sensitivity solver."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.sparse import csc_array

from pyrtid.forward.flow_solver import solve_fl_gmres
from pyrtid.forward.models import ForwardModel, Geometry
from pyrtid.forward.solver import ForwardSolver
from pyrtid.inverse.fsm.dFds import (
    dFcdcimp,
    dFcdmimp,
    dFcdwv,
    dFDdav,
    dFDdDv,
    dFDdwv,
    dFhdhimp,
    dFhdKv,
    dFhdSsv,
    dFmdcimp,
    dFmdmimp,
    dFpdKv,
    dFpdpimp,
    dFpdSsv,
    dFUxdKv,
    dFUydKv,
)
from pyrtid.inverse.obs import (
    Observables,
    get_observables_values_as_1d_vector,
    get_predictions_matching_observations,
)
from pyrtid.inverse.params import AdjustableParameters, ParameterName
from pyrtid.utils import NDArrayFloat, object_or_object_sequence_to_list


class FSMVects:
    """
    Class to hold the vectors defining dFdSv and dFdSv for the FSM.

    Since most vectors will be null, we rely on sparse objects to save some memory.
    """

    def __init__(self, geometry: Geometry, vecs: NDArrayFloat, n_obs: int) -> None:
        """Initiate the instance."""

        self.vecs = vecs
        # number of adjusted values and number of rhs vectors to be multiplied with the
        # Jacobian matrix.
        ns, ne = vecs.shape

        # create sparse arrays
        self.dFhdsv = csc_array((geometry.nx, geometry.ny))
        self.dFpdsv = csc_array((geometry.nx, geometry.ny))
        self.dFUxdsv = csc_array((geometry.nx + 1, geometry.ny))
        self.dFUydsv = csc_array((geometry.nx, geometry.ny + 1))
        self.dFDdsv = csc_array((geometry.nx, geometry.ny))
        self.dFcdsv = csc_array((2, geometry.nx, geometry.ny))
        self.dFmdsv = csc_array((2, geometry.nx, geometry.ny))
        self.dFrhodsv = csc_array((geometry.nx, geometry.ny))

        self.zh = csc_array((geometry.nx, geometry.ny, ne))
        self.zp = csc_array((geometry.nx, geometry.ny, ne))
        self.zUx = csc_array((geometry.nx + 1, geometry.ny, ne))
        self.zUy = csc_array((geometry.nx, geometry.ny + 1, ne))
        self.zD = csc_array((geometry.nx, geometry.ny, ne))
        self.zc = csc_array((2, geometry.nx, geometry.ny, ne))
        self.zm = csc_array((2, geometry.nx, geometry.ny, ne))
        self.zrho = csc_array((geometry.nx, geometry.ny, ne))

        # Array in which the Ne products with the Jacobian matrix are stored.
        # This matrix has shape (N_obs, Ne)
        self.jacvecs: NDArrayFloat = np.zeros(
            (n_obs, ne),
        )

    def clear_dFdsv(self) -> None:
        """Reinitialize the arrays storing dFdsv to zero."""
        self.dFhdsv[self.dFhdsv.nonzero()] = 0.0
        self.dFpdsv[self.dFpdsv.nonzero()] = 0.0
        self.dFUxdsv[self.dFUxdsv.nonzero()] = 0.0
        self.dFUydsv[self.dFUydsv.nonzero()] = 0.0
        self.dFDdsv[self.dFDdsv.nonzero()] = 0.0
        self.dFcdsv[self.dFcdsv.nonzero()] = 0.0
        self.dFmdsv[self.dFmdsv.nonzero()] = 0.0
        self.dFrhodsv[self.dFrhodsv.nonzero()] = 0.0


class FSMSolver:
    """Class solving the reactive transport forward systems."""

    def __init__(self, model: ForwardModel) -> None:
        # The model needs to be copied
        self.model: ForwardModel = model
        self.solver = ForwardSolver(model)

    def solve(
        self,
        observables: Observables,
        parameters_to_adjust: AdjustableParameters,
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

        # TODO: Apply preconditioning to vects -> subsampling etc.

        # Initiate an instance to hold the FSM temp vectors
        fsm_vects = FSMVects(self.model.geometry, vecs, n_obs)

        # compute the sensitivities for t=0
        # z is the working vector
        self.solve_sensitivities(parameters_to_adjust, fsm_vects, time_index)

        # Fill jacvecs: NDArrayFloat

        # Sequential iterative approach with operator splitting
        while self.model.time_params.time_elapsed < self.model.time_params.duration:
            time_index += 1  # Update the number of time iterations
            # Reset numerical acceleration if it was temporarily disabled
            self.model.tr_model.is_num_acc_for_timestep = (
                self.model.tr_model.is_numerical_acceleration
            )
            self.solver._solve_system_for_timestep(time_index, is_verbose)

            self.solve_sensitivities(parameters_to_adjust, fsm_vects, time_index)

        # get the predictions
        d_pred = get_predictions_matching_observations(
            self.model, observables, hm_end_time
        )

        return d_pred, fsm_vects.jacvecs

    def solve_sensitivities(
        self,
        parameters_to_adjust: AdjustableParameters,
        fsm_vecs: FSMVects,
        time_index: int,
    ) -> None:
        # step 1: compute - (d F^n(s, u) / d s) @ v
        # note that this is the same as computing the derivative of
        # d < \lambda_u, F^n(s, u) > / ds
        # So we can reuse the code for the adjoint state, replacing the adjoint variable
        # with the vector we are interested in.
        update_dFdsv(self.model, parameters_to_adjust, fsm_vecs, time_index)

        # step 2: compute ...
        # For this step, we first treat the flow to handle saturated vs density
        self.solve_sensitities_flow(fsm_vecs, time_index)

        # Then the transport is the same in both cases
        # TODO

    def solve_sensitities_flow(self, fsm_vecs: FSMVects, time_index: int) -> None:
        if self.model.fl_model.is_gravity:
            # solve on pressure
            rhs = -fsm_vecs.dFpdsv
            # reference to the correct z
            z = fsm_vecs.zp
        else:
            # solve on head
            rhs = -fsm_vecs.dFhdsv
            # reference to the correct z
            z = fsm_vecs.zh

        # add -B^{n-1} z^{n-1}
        rhs -= self.model.fl_model.q_prev @ z

        # Solve A^n z^n = rhs
        for i in range(z.shape[1]):
            z[:, i], exit_code = solve_fl_gmres(
                self.model.fl_model,
                rhs[:, i],
                self.model.fl_model.super_ilu,
                self.model.fl_model.preconditioner,
            )

        # TODO: check if z is correctly updated in fsm_vecs

        # # Now update the second variables (h or p depending on the case.)
        # if self.model.fl_model.is_gravity:
        #     # update of the head from the pressure -> identity matrix
        #     # to check dp/dh @ z
        #     rhs = -fsm_vecs.dFhdsv
        #     # update of the pressure from the head -> identity matrix as well
        #     z = fsm_vecs.zh
        # else:
        #     # solve on head
        #     rhs = -fsm_vecs.dFhdsv
        #     # reference to the correct z
        #     z = fsm_vecs.zp

        # z = rhs

        # # Finally, deal with the darcy velocities
        # rhs = -fsm_vecs.dFUdsv
        # # reference to the correct z
        # fsm_vecs.zUx, fsm_vecs.zUy = ...

    def fill_jacvecs(
        self, jacvecs: NDArrayFloat, z_tmp: NDArrayFloat, time_index: int
    ) -> None:
        # Derivative of the observations with respect to h^{n} @ z_tmp
        # For now do nothing
        jacvecs[:, :] = jacvecs[:, :]

        # TODO: this is currently the missing step

        # We have to build a system to iterate the observations.


def update_dFdsv(
    model: ForwardModel,
    parameters_to_adjust: AdjustableParameters,
    fsm_vecs: FSMVects,
    time_index: int,
) -> None:
    # set all to zero
    fsm_vecs.clear_dFdsv()

    # TODO: the first step is to finish the implementation of this.

    # loop over parameters
    for param in object_or_object_sequence_to_list(parameters_to_adjust):
        if param.name == ParameterName.PERMEABILITY:
            fsm_vecs.dFhdsv += dFhdKv(model, time_index, fsm_vecs.vecs)
            fsm_vecs.dFpdsv += dFpdKv(model, time_index, fsm_vecs.vecs)
            fsm_vecs.dFUxdsv += dFUxdKv(model, time_index, fsm_vecs.vecs)
            fsm_vecs.dFUydsv += dFUydKv(model, time_index, fsm_vecs.vecs)
        if param.name == ParameterName.STORAGE_COEFFICIENT:
            fsm_vecs.dFhdsv += dFhdSsv(model, time_index, fsm_vecs.vecs)
            fsm_vecs.dFpdsv += dFpdSsv(model, time_index, fsm_vecs.vecs)
        if param.name == ParameterName.INITIAL_HEAD:
            fsm_vecs.dFhdsv += dFhdhimp(model, time_index, fsm_vecs.vecs)
        if param.name == ParameterName.INITIAL_PRESSURE:
            fsm_vecs.dFpdsv += dFpdpimp(model, time_index, fsm_vecs.vecs)
        if param.name == ParameterName.POROSITY:
            fsm_vecs.dFDdsv += dFDdwv(model, time_index, fsm_vecs.vecs)
            fsm_vecs.dFcdsv += dFcdwv(model, time_index, fsm_vecs.vecs)
        if param.name == ParameterName.DIFFUSION:
            fsm_vecs.dFDdsv += dFDdDv(model, time_index, fsm_vecs.vecs)
        if param.name == ParameterName.DISPERSIVITY:
            fsm_vecs.dFDdsv += dFDdav(model, time_index, fsm_vecs.vecs)
        if param.name == ParameterName.INITIAL_CONCENTRATION:
            fsm_vecs.dFcdsv += dFcdcimp(model, time_index, fsm_vecs.vecs)
            fsm_vecs.dFmdsv += dFmdcimp(model, time_index, fsm_vecs.vecs)
        if param.name == ParameterName.INITIAL_GRADE:
            fsm_vecs.dFcdsv += dFcdmimp(model, time_index, fsm_vecs.vecs)
            fsm_vecs.dFmdsv += dFmdmimp(model, time_index, fsm_vecs.vecs)
