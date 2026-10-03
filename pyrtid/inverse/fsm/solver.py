# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 Antoine COLLET

r"""
Provide a forward sensitivity solver.

The forward sensitivity method (FSM) computes the product of the Jacobian matrix
:math:`\mathbf{J} = \partial \mathbf{d}_{\mathrm{pred}} / \partial \mathbf{s}` of the
predictions with a set of vectors :math:`\mathbf{V}` without building the matrix. The
forward model is solved, and, at each timestep, the sensitivities :math:`z` of the
state variables to the perturbation :math:`\mathbf{V}` are propagated:

.. math::
    z^n = \left(\frac{\partial F^n}{\partial u^n}\right)^{-1}
    \left(- \frac{\partial F^n}{\partial u^{n-1}} z^{n-1}
    - \frac{\partial F^n}{\partial s} \mathbf{V} \right)

The following equations are considered (see :mod:`pyrtid.inverse.fsm.dFds` and
:mod:`pyrtid.inverse.fsm.dFdu`):

1. the flow (head), solved with the matrices of the forward model;
2. the darcy velocities, their divergence and the unit flows, which are linear
   operators on the head;
3. the transport of the mobile species coupled to the chemistry (the grades). The
   sensitivities of the coupled system are solved in one go, which corresponds to
   the limit of the fixed point iterations of the forward solver.

Only the saturated flow is supported (the gravity must be off).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from inv_toolbox.utils.means import get_mean_values_gradient_for_last_axis
from quickpaver import rlg_nn_to_idx
from scipy.sparse import csc_array
from scipy.sparse.linalg import splu

from pyrtid.forward.flow_solver import GRAVITY, WATER_DENSITY
from pyrtid.forward.models import TDS_LINEAR_COEFFICIENT, FlowRegime, ForwardModel
from pyrtid.forward.solver import ForwardSolver
from pyrtid.inverse.fsm.dFds import (
    FlowFields,
    dFcds,
    dFhds_stationary,
    dFhds_transient,
    dflow_fields,
    get_flow_sources,
)
from pyrtid.inverse.fsm.dFdu import get_chemistry_relations, get_coupled_matrix
from pyrtid.inverse.fsm.directions import FSMDirections, get_directions
from pyrtid.inverse.obs import (
    Observable,
    Observables,
    StateVariable,
    get_array_from_state_variable,
    get_predictions_matching_observations,
    get_sorted_observable_times,
    get_times_idx_before_after_obs,
    get_weights,
)
from pyrtid.inverse.params import AdjustableParameters
from pyrtid.utils import NDArrayFloat, object_or_object_sequence_to_list

logger = logging.getLogger(__name__)


@dataclass
class FSMState:
    """
    Sensitivities of the state variables at one time.

    All the arrays have a trailing dimension of size ``ne`` (the number of vectors).

    Attributes
    ----------
    head : NDArrayFloat
        Sensitivity of the head with shape (nx * ny * nz, ne).
    mob : NDArrayFloat
        Sensitivity of the mobile concentrations with shape (2, nx * ny * nz, ne).
    immob : NDArrayFloat
        Sensitivity of the grades with shape (2, nx * ny * nz, ne).
    fields : FlowFields
        Sensitivity of the flow fields (darcy velocities, divergence and unit flow).
    """

    head: NDArrayFloat
    mob: NDArrayFloat
    immob: NDArrayFloat
    fields: FlowFields


class FSMObservationRecorder:
    """
    Accumulate the products of the observation operators with the sensitivities.

    For each observable, the sensitivity of the spatial average of the state
    variable is stored at each simulation time. At the end of the simulation, they
    are interpolated in time to match the observation times.
    """

    def __init__(
        self,
        model: ForwardModel,
        observables: Observables,
        directions: FSMDirections,
        hm_end_time: float | None,
    ) -> None:
        """
        Initialize the instance.

        Parameters
        ----------
        model : ForwardModel
            The forward model.
        observables : Observables
            The observables.
        directions : FSMDirections
            The perturbations of the parameters.
        hm_end_time : float | None
            Observations after this time are ignored.
        """
        self.model = model
        self.observables: list[Observable] = object_or_object_sequence_to_list(
            observables
        )
        self.directions = directions
        self.hm_end_time = hm_end_time
        self.ne = directions.ne
        # sensitivities of the averaged state variable for each time index
        self.history: list[list[NDArrayFloat]] = [[] for _ in self.observables]
        self.constants: dict[int, NDArrayFloat] = {}

    def _get_parameter_direction(self, obs: Observable) -> NDArrayFloat | None:
        """Return the perturbation of an observed parameter (None if not adjusted)."""
        mapping = {
            StateVariable.PERMEABILITY: self.directions.permeability,
            StateVariable.STORAGE_COEFFICIENT: self.directions.storage_coefficient,
            StateVariable.POROSITY: self.directions.porosity,
            StateVariable.DIFFUSION: self.directions.diffusion,
            StateVariable.DISPERSIVITY: self.directions.dispersivity,
        }
        return mapping[obs.state_variable]

    def _get_state_sensitivity(self, obs: Observable, state: FSMState) -> NDArrayFloat:
        """Return the sensitivity of the state variable with shape (n_cells, ne)."""
        var = obs.state_variable
        if var == StateVariable.HEAD:
            return state.head
        if var == StateVariable.PRESSURE:
            return GRAVITY * WATER_DENSITY * state.head
        if var == StateVariable.CONCENTRATION:
            return state.mob[obs.sp]
        if var == StateVariable.GRADE:
            return state.immob[obs.sp]
        # density
        gch = self.model.gch_params
        return (
            WATER_DENSITY
            * TDS_LINEAR_COEFFICIENT
            * (state.mob[0] * gch.Ms + state.mob[1] * gch.Ms2)
            / 1000.0
        )

    def _get_field_values(self, obs: Observable, time_index: int) -> NDArrayFloat:
        """Return the field of the state variable at a given time (flat)."""
        var = obs.state_variable
        fl_model = self.model.fl_model
        tr_model = self.model.tr_model
        if var == StateVariable.HEAD:
            arr = fl_model.lhead[time_index]
        elif var == StateVariable.PRESSURE:
            arr = fl_model.lpressure[time_index]
        elif var == StateVariable.CONCENTRATION:
            arr = tr_model.lmob[time_index][obs.sp]
        elif var == StateVariable.GRADE:
            arr = tr_model.limmob[time_index][obs.sp]
        else:
            arr = tr_model.ldensity[time_index]
        return arr.ravel(order="F")

    def record(self, time_index: int, state: FSMState) -> None:
        """Store the sensitivity of the averaged state variables at a given time."""
        for i, obs in enumerate(self.observables):
            if obs.state_variable in (
                StateVariable.PERMEABILITY,
                StateVariable.STORAGE_COEFFICIENT,
                StateVariable.POROSITY,
                StateVariable.DIFFUSION,
                StateVariable.DISPERSIVITY,
            ):
                continue
            sens = self._get_state_sensitivity(obs, state)
            values = self._get_field_values(obs, time_index)[obs.node_indices]
            grad = get_mean_values_gradient_for_last_axis(
                values.reshape(-1, 1), mean_type=obs.mean_type, weights=None
            )
            self.history[i].append(grad[:, 0] @ sens[obs.node_indices, :])

    def _parameter_constant(self, obs: Observable) -> NDArrayFloat:
        """Return the sensitivity of the average of an observed parameter."""
        direction = self._get_parameter_direction(obs)
        if direction is None:
            return np.zeros(self.ne)
        field = get_array_from_state_variable(self.model, obs.state_variable, obs.sp)
        X, Y, Z = rlg_nn_to_idx(
            obs.node_indices, nx=self.model.grid.nx, ny=self.model.grid.ny
        )
        grad = get_mean_values_gradient_for_last_axis(
            field[X, Y, Z].reshape(-1, 1), mean_type=obs.mean_type, weights=None
        )
        return grad[:, 0] @ direction[X, Y, Z, :]

    def get_jacvecs(self) -> NDArrayFloat:
        """
        Return the products of the Jacobian matrix with the vectors.

        Returns
        -------
        NDArrayFloat
            Array with shape (:math:`N_{obs}`, :math:`N_e`).
        """
        simu_times = np.cumsum([0, *self.model.time_params.ldt])
        max_obs_time = (
            simu_times.max()
            if self.hm_end_time is None
            else min(simu_times.max(), self.hm_end_time)
        )
        out = []
        for i, obs in enumerate(self.observables):
            obs_times = get_sorted_observable_times(obs, max_obs_time)
            if obs.state_variable in (
                StateVariable.PERMEABILITY,
                StateVariable.STORAGE_COEFFICIENT,
                StateVariable.POROSITY,
                StateVariable.DIFFUSION,
                StateVariable.DISPERSIVITY,
            ):
                out.append(np.tile(self._parameter_constant(obs), (obs_times.size, 1)))
                continue
            sens = np.array(self.history[i])  # (nt + 1, ne)
            before, after = get_times_idx_before_after_obs(obs_times, simu_times)
            w_before, w_after = get_weights(obs_times, simu_times, before, after)
            out.append(
                w_before[:, np.newaxis] * sens[before]
                + w_after[:, np.newaxis] * sens[after]
            )
        return np.vstack(out) if out else np.zeros((0, self.ne))


class FSMSolver:
    """Solve the reactive transport forward system and its forward sensitivities."""

    def __init__(self, model: ForwardModel) -> None:
        """
        Initialize the instance.

        Parameters
        ----------
        model : ForwardModel
            The forward model. It is solved (and thus modified) by :meth:`solve`.
        """
        self.model: ForwardModel = model
        self.solver = ForwardSolver(model)

    def solve(
        self,
        observables: Observables,
        parameters_to_adjust: AdjustableParameters,
        vecs: NDArrayFloat,
        hm_end_time: float | None = None,
        is_verbose: bool = False,
    ) -> tuple[NDArrayFloat, NDArrayFloat]:
        r"""
        Solve the forward problem and apply the forward sensitivity method.

        Parameters
        ----------
        observables : Observables
            The observables.
        parameters_to_adjust : AdjustableParameters
            The adjusted parameters, which define the rows of ``vecs``.
        vecs : NDArrayFloat
            Vectors to multiply with the Jacobian matrix of the predictions with
            respect to the (preconditioned) adjusted parameters. It has shape
            (:math:`N_s`, :math:`N_e`), :math:`N_s` being the number of adjusted
            values and :math:`N_e` the number of vectors.
        hm_end_time : float | None, optional
            Observations after this time are ignored. The default is None.
        is_verbose : bool, optional
            Whether to display info. The default is False.

        Returns
        -------
        tuple[NDArrayFloat, NDArrayFloat]
            The predictions matching the observations, with shape (:math:`N_{obs}`,),
            and the products between the Jacobian matrix and the vectors, with shape
            (:math:`N_{obs}`, :math:`N_e`).

        Raises
        ------
        NotImplementedError
            If the model uses a density flow (the gravity is on).
        """
        if self.model.fl_model.is_gravity:
            raise NotImplementedError(
                "The forward sensitivity method is not implemented for density flows."
            )
        directions = get_directions(self.model, parameters_to_adjust, vecs)
        recorder = FSMObservationRecorder(
            self.model, observables, directions, hm_end_time
        )

        # solve for t=0
        self.solver.initialize()
        time_index = 0
        state = self._initial_state(directions)
        recorder.record(time_index, state)

        # Sequential iterative approach with operator splitting
        time_params = self.model.time_params
        while time_params.time_elapsed < time_params.duration:
            time_index += 1
            # Reset numerical acceleration if it was temporarily disabled
            self.model.tr_model.is_num_acc_for_timestep = (
                self.model.tr_model.is_numerical_acceleration
            )
            self.solver._solve_system_for_timestep(time_index, is_verbose)
            state = self._solve_sensitivities(state, directions, time_index)
            recorder.record(time_index, state)

        d_pred = get_predictions_matching_observations(
            self.model, observables, hm_end_time
        )
        return d_pred, recorder.get_jacvecs()

    def _initial_state(self, directions: FSMDirections) -> FSMState:
        """Return the sensitivities at the initial time."""
        model = self.model
        grid = model.grid
        fl_model = model.fl_model
        n_cells = grid.n_grid_cells
        ne = directions.ne

        # initial head
        d_head = np.zeros((n_cells, ne))
        if directions.head is not None:
            d_head = directions.head.reshape(n_cells, ne, order="F")

        if fl_model.regime == FlowRegime.STATIONARY:
            # The stationary flow is solved with the initial head at the constant
            # head nodes.
            matrix = splu(fl_model.q_next.tocsc())
            for e in range(ne):
                rhs = dFhds_stationary(model, directions.column(e, "permeability"))
                rhs[fl_model.cst_head_nn] = d_head[fl_model.cst_head_nn, e]
                d_head[:, e] = matrix.solve(rhs)

        # sensitivities of the fields at t=0
        if fl_model.regime == FlowRegime.STATIONARY:
            fields = self._get_fields_sensitivity(0, d_head, directions)
        else:  # null velocities
            fields = FlowFields.zeros(grid, ne)

        mob = np.zeros((2, n_cells, ne))
        immob = np.zeros((2, n_cells, ne))
        for sp in range(2):
            conc_dir = directions.conc[sp]
            if conc_dir is not None:
                mob[sp] = conc_dir.reshape(n_cells, ne, order="F")
            grade_dir = directions.grade[sp]
            if grade_dir is not None:
                immob[sp] = grade_dir.reshape(n_cells, ne, order="F")
        return FSMState(d_head, mob, immob, fields)

    def _get_fields_sensitivity(
        self, time_index: int, d_head: NDArrayFloat, directions: FSMDirections
    ) -> FlowFields:
        """Return the sensitivities of the flow fields in all the directions."""
        grid = self.model.grid
        ne = directions.ne
        out = FlowFields.zeros(grid, ne)
        unitflow = get_flow_sources(self.model, time_index)
        for e in range(ne):
            out.set_column(
                e,
                dflow_fields(
                    self.model,
                    time_index,
                    unitflow,
                    directions.column(e, "permeability"),
                    d_head[:, e].reshape(grid.shape, order="F"),
                ),
            )
        return out

    def _solve_sensitivities(
        self, prev: FSMState, directions: FSMDirections, time_index: int
    ) -> FSMState:
        """Return the sensitivities at ``time_index`` from the ones at the previous."""
        dt = self.model.time_params.ldt[time_index - 1]
        d_head = self._solve_flow_sensitivities(prev, directions, time_index, dt)
        fields = self._get_fields_sensitivity(time_index, d_head, directions)
        mob, immob = self._solve_transport_sensitivities(
            prev, fields, directions, time_index, dt
        )
        return FSMState(d_head, mob, immob, fields)

    def _solve_flow_sensitivities(
        self, prev: FSMState, directions: FSMDirections, time_index: int, dt: float
    ) -> NDArrayFloat:
        """Return the head sensitivities of a timestep, with shape (n_cells, ne)."""
        fl_model = self.model.fl_model
        ne = directions.ne
        matrix = splu(fl_model.q_next.tocsc())
        rhs = fl_model.q_prev @ prev.head
        for e in range(ne):
            rhs[:, e] += dFhds_transient(
                self.model,
                time_index,
                dt,
                directions.column(e, "permeability"),
                directions.column(e, "storage_coefficient"),
            )
        # constant head grid cells
        rhs[fl_model.cst_head_nn, :] = prev.head[fl_model.cst_head_nn, :]
        return matrix.solve(rhs)

    def _solve_transport_sensitivities(
        self,
        prev: FSMState,
        fields: FlowFields,
        directions: FSMDirections,
        time_index: int,
        dt: float,
    ) -> tuple[NDArrayFloat, NDArrayFloat]:
        """Return the sensitivities of the concentrations and grades of a timestep."""
        model = self.model
        tr_model = model.tr_model
        if tr_model.is_skip_rt:  # nothing is computed: the concentrations are constant
            return prev.mob.copy(), prev.immob.copy()
        fl_model = model.fl_model
        n_cells = model.grid.n_grid_cells
        ne = directions.ne

        fields_old = FlowFields.from_model(fl_model, time_index - 1)
        fields_new = FlowFields.from_model(fl_model, time_index)

        # forcing terms
        forcing = np.zeros((2, n_cells, ne))
        for e in range(ne):
            forcing[:, :, e] = dFcds(
                model,
                time_index,
                dt,
                fields_old,
                fields_new,
                prev.fields.column(e),
                fields.column(e),
                directions.column(e, "porosity"),
                directions.column(e, "diffusion"),
                directions.column(e, "dispersivity"),
            )

        # operators
        porosity_over_dt = tr_model.porosity.ravel(order="F") / dt
        l_mob, l_immob, l_prev = get_chemistry_relations(model, time_index, dt)

        # right hand side
        # - transport: Q^{n-1} z^{n-1} + w/dt zbar^{n-1} - forcing
        # - chemistry: L_p zbar^{n-1}
        rhs = np.zeros((4, n_cells, ne))
        for i in range(2):
            rhs[i] = (
                tr_model.q_prev @ prev.mob[i]
                + porosity_over_dt[:, np.newaxis] * prev.immob[i]
                - forcing[i]
            )
            for j in range(2):
                rhs[2 + i] += l_prev[i, j][:, np.newaxis] * prev.immob[j]

        matrix = splu(
            get_coupled_matrix(
                csc_array(tr_model.q_next), porosity_over_dt, l_mob, l_immob
            )
        )
        sol = matrix.solve(rhs.reshape(4 * n_cells, ne)).reshape(4, n_cells, ne)
        return sol[:2], sol[2:]
