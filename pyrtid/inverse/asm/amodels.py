"""Provide the models for the adjoint states."""

from __future__ import annotations

import copy
from abc import ABC
from typing import List, Optional

import numpy as np
from scipy.sparse import csc_array, lil_array

from pyrtid.forward.models import (  # ConstantHead,; ZeroConcGradient,
    ForwardModel,
    Geometry,
    TimeParameters,
)
from pyrtid.inverse.obs import (
    Observables,
    StateVariable,
    get_adjoint_sources_for_obs,
    get_observables_values_as_1d_vector,
)
from pyrtid.utils import object_or_object_sequence_to_list
from pyrtid.utils.types import NDArrayFloat


class AdjointFlowModel(ABC):
    """Represent an adjoint flow model."""

    __slots__ = [
        "a_head",
        "a_pressure",
        "a_u_darcy_x",
        "a_u_darcy_y",
        "a_head_sources",
        "a_pressure_sources",
        "a_permeability_sources",
        "a_storage_coefficient_sources",
        "q_prev",
        "q_next",
        "crank_nicolson",
        "rtol",
        "is_use_continuous_adj",
        "l_q_prev",
        "l_q_next",
    ]

    def __init__(
        self,
        grid: Geometry,
        time_params: TimeParameters,
        is_use_continuous_adj: bool = False,
    ) -> None:
        """
        Initialize the instance.

        Parameters
        ----------
        grid : Geometry
            Geometry of the problem, grid definition.
        time_params : TimeParameters
            Time parameters from the forward problem.
        is_use_continuous_adj: bool
            Whether to use the adjoint state derived using the "continuous" way,
            aka the differentiate-then-discretize method.
            This is only working with saturated flow. This option has been
            added to illustrate the paper TODO: add ref. The default is False.
        """
        self.a_head = np.zeros((grid.nx, grid.ny, time_params.nt), dtype=np.float64)
        self.a_pressure = np.zeros((grid.nx, grid.ny, time_params.nt), dtype=np.float64)
        self.a_u_darcy_x = np.zeros(
            (grid.nx + 1, grid.ny, time_params.nt), dtype=np.float64
        )
        self.a_u_darcy_y = np.zeros(
            (grid.nx, grid.ny + 1, time_params.nt), dtype=np.float64
        )
        # Generally, not so many observations, so only a few adjoint variable,
        # so use a sparse matrix instead of a dense array
        # NOTE: rows are grid cells, and columns are time indices
        # We use csc format for fast column (time) slicing
        self.a_head_sources: csc_array = csc_array(
            (grid.nx * grid.ny, time_params.nt), dtype=np.float64
        )
        self.a_pressure_sources: csc_array = csc_array(
            (grid.nx * grid.ny, time_params.nt), dtype=np.float64
        )
        # does not vary in time
        self.a_permeability_sources: csc_array = csc_array(
            (grid.nx * grid.ny, 1), dtype=np.float64
        )
        self.a_storage_coefficient_sources: csc_array = csc_array(
            (grid.nx * grid.ny, 1), dtype=np.float64
        )

        self.q_prev: lil_array = lil_array((grid.nx * grid.ny, 1))
        self.q_next: lil_array = lil_array((grid.nx * grid.ny, 1))

        # crank nicolson: if None, then the crank-nicolson from the forward model
        # is used. This attribute only purpose is to test the impact of an
        # incorrect discretization.
        self.crank_nicolson: Optional[float] = None
        self.rtol: float = 1e-8
        self.is_use_continuous_adj: bool = is_use_continuous_adj

        # List to store the successive stiffness matrices
        # This is mostly for development purposes.
        self.l_q_next: List[lil_array] = []
        self.l_q_prev: List[lil_array] = []

    def clear_adjoint_sources(self) -> None:
        """
        Reset all adjoint sources to zero.
        """
        self.a_head_sources = csc_array(self.a_head_sources.shape)
        self.a_pressure_sources = csc_array(self.a_pressure_sources.shape)
        self.a_permeability_sources = csc_array(self.a_permeability_sources.shape)
        self.a_storage_coefficient_sources = csc_array(
            self.a_storage_coefficient_sources.shape
        )

    def set_crank_nicolson(self, value: Optional[float]) -> None:
        self.crank_nicolson = value

    def reinit(self) -> None:
        """Reinitialize the adjoint flow model. Set all arrays to zero."""
        self.clear_adjoint_sources()
        self.a_head = np.zeros_like(self.a_head)
        self.a_pressure = np.zeros_like(self.a_pressure)
        self.a_u_darcy_x = np.zeros_like(self.a_u_darcy_x)
        self.a_u_darcy_y = np.zeros_like(self.a_u_darcy_y)
        self.l_q_next = []
        self.l_q_prev = []


class SaturatedAdjointFlowModel(AdjointFlowModel):
    """
    Saturated Adjoint Flow Model.
    """

    def __init__(
        self,
        grid: Geometry,
        time_params: TimeParameters,
        is_use_continuous_adj: bool = False,
    ) -> None:
        """
        Initialize the instance.

        Parameters
        ----------
        grid : Geometry
            Geometry of the problem, grid definition.
        time_params : TimeParameters
            Time parameters from the forward problem.
        is_use_continuous_adj: bool
            Whether to use the adjoint state derived using the "continuous" way,
            aka the differentiate-then-discretize method.
            This is only working with saturated flow. This option has been
            added to illustrate the paper TODO: add ref. The default is False.

        """
        super().__init__(grid, time_params, is_use_continuous_adj)


class DensityAdjointFlowModel(AdjointFlowModel):
    """
    Density Adjoint Flow Model.
    """

    def __init__(
        self,
        grid: Geometry,
        time_params: TimeParameters,
        is_use_continuous_adj: bool = False,
    ) -> None:
        """
        Initialize the instance.

        Parameters
        ----------
        grid : Geometry
            Geometry of the problem, grid definition.
        time_params : TimeParameters
            Time parameters from the forward problem.
        is_use_continuous_adj: bool
            Whether to use the adjoint state derived using the "continuous" way,
            aka the differentiate-then-discretize method.
            This is only working with saturated flow. This option has been
            added to illustrate the paper TODO: add ref. The default is False.

        """
        if is_use_continuous_adj:
            raise ValueError("Continuous adjoint not implemented for density flow!")
        super().__init__(grid, time_params, is_use_continuous_adj)


class AdjointTransportModel:
    """
    Represent an adjoint flow model.

    Attributes
    ----------
    a_mob: NDArrayFloat
    a_mob_prev: NDArrayFloat
    a_immob: NDArrayFloat
    a_conc_sources: NDArrayFloat
    a_conc_2_sources: NDArrayFloat
    a_grade_sources: NDArrayFloat
    q_prev: lil_array
    q_next: lil_array
    a_gch_src_term: NDArrayFloat
    afpi_eps: float
    is_adj_numerical_acceleration: bool
    is_adj_num_acc_for_timestep: bool
    n_sp: int
    """

    __slots__ = [
        "a_mob",
        "a_mob_prev",
        "a_immob",
        "a_density",
        "a_conc_sources",
        "a_grade_sources",
        "a_porosity_sources",
        "a_diffusion_sources",
        "a_dispersivity_sources",
        "a_density_sources",
        "q_prev",
        "q_next",
        "a_gch_src_term",
        "afpi_eps",
        "is_adj_numerical_acceleration",
        "is_adj_num_acc_for_timestep",
        "n_sp",
        "l_q_prev",
        "l_q_next",
    ]

    def __init__(
        self,
        grid: Geometry,
        time_params: TimeParameters,
        n_sp: int,
        afpi_eps: float = 1e-5,
        is_adj_numerical_acceleration: bool = False,
    ) -> None:
        """
        Initialize the instance.

        Parameters
        ----------
        grid: Geometry
            Simulation grid definition.
        time_params: TimeParameters
            Simulation time parameters (duration, timesteps, etc.)
        n_sp: int
            Number of species in the system.
        afpi_eps: float
            Adjoint fixed point iteration criterion.
        is_numerical_acceleration: bool
            Whether to use numerical acceleration in the ajdoint.

        """
        self.n_sp = n_sp
        self.a_mob: NDArrayFloat = np.zeros(
            (self.n_sp, grid.nx, grid.ny, time_params.nt), dtype=np.float64
        )
        self.a_mob_prev: NDArrayFloat = np.zeros(
            (self.n_sp, grid.nx, grid.ny), dtype=np.float64
        )

        self.a_immob: NDArrayFloat = np.zeros(
            (self.n_sp, grid.nx, grid.ny, time_params.nt), dtype=np.float64
        )
        self.a_density: NDArrayFloat = np.zeros(
            (grid.nx, grid.ny, time_params.nt), dtype=np.float64
        )

        # Generally, not so many observations, so only a few adjoint variable,
        # so use a sparse matrix instead of a dense array
        # NOTE: rows are grid cells, and columns are time indices
        # We use csc format for fast column (time) slicing
        self.a_conc_sources: List[csc_array] = [
            csc_array((grid.nx * grid.ny, time_params.nt), dtype=np.float64)
            for sp in range(self.n_sp)  # type: ignore
        ]
        self.a_grade_sources: List[csc_array] = copy.copy(self.a_conc_sources)
        self.a_porosity_sources = csc_array((grid.nx * grid.ny, 1), dtype=np.float64)
        self.a_diffusion_sources = csc_array((grid.nx * grid.ny, 1), dtype=np.float64)
        self.a_dispersivity_sources = csc_array(
            (grid.nx * grid.ny, 1), dtype=np.float64
        )
        self.a_density_sources: csc_array = csc_array(
            (grid.nx * grid.ny, time_params.nt), dtype=np.float64
        )

        # Adjoint source term from the adjoint geochem to the adjoint transport
        self.a_gch_src_term = np.zeros((n_sp, grid.nx, grid.ny), dtype=np.float64)

        self.q_prev: lil_array = lil_array((grid.nx * grid.ny, 1))
        self.q_next: lil_array = lil_array((grid.nx * grid.ny, 1))
        self.afpi_eps = afpi_eps
        self.is_adj_numerical_acceleration: bool = is_adj_numerical_acceleration
        self.is_adj_num_acc_for_timestep: bool = self.is_adj_numerical_acceleration

    def clear_adjoint_sources(self) -> None:
        """Reset all adjoint sources to zero."""
        self.a_conc_sources = [
            csc_array(self.a_grade_sources[0].shape) for sp in range(self.n_sp)
        ]
        self.a_grade_sources = copy.copy(self.a_conc_sources)
        self.a_porosity_sources = csc_array(self.a_porosity_sources.shape)
        self.a_diffusion_sources = csc_array(self.a_diffusion_sources.shape)
        self.a_dispersivity_sources = csc_array(self.a_dispersivity_sources.shape)
        self.a_density_sources = csc_array(self.a_density_sources.shape)

        # List to store the successive stiffness matrices
        # This is mostly for development purposes.
        self.l_q_next: List[lil_array] = []
        self.l_q_prev: List[lil_array] = []

    @property
    def a_conc(self) -> NDArrayFloat:
        """Alias for a_mob."""
        return self.a_mob

    @property
    def a_grade(self) -> NDArrayFloat:
        """Alias for a_immob."""
        return self.a_immob

    def reinit(self) -> None:
        """Reinitialize the adjoint transport model. Set all arrays to zero."""
        self.clear_adjoint_sources()
        self.a_mob = np.zeros_like(self.a_mob)
        self.a_mob_prev = np.zeros_like(self.a_mob_prev)
        self.a_immob = np.zeros_like(self.a_immob)
        self.a_density = np.zeros_like(self.a_density)
        self.a_gch_src_term = np.zeros_like(self.a_gch_src_term)
        self.l_q_next = []
        self.l_q_prev = []


class AdjointModel:
    """Represent an adjoint model."""

    __slots__ = ["grid", "time_params", "gch_params", "a_fl_model", "a_tr_model"]

    def __init__(
        self,
        grid: Geometry,
        time_params: TimeParameters,
        is_gravity: bool,
        n_sp: int,
        afpi_eps: float = 1e-5,
        is_adj_numerical_acceleration: bool = False,
        is_use_continuous_adj: bool = False,
    ) -> None:
        """
        Initialize the instance.

        Parameters
        ----------
        grid: Geometry
            Simulation grid definition.
        time_params: TimeParameters
            Simulation time parameters (duration, timesteps, etc.)
        is_gravity: bool
            Whether to consider gravity for a density driven flow.
        n_sp: int
            The number of species in the system.
        afpi_eps: float
            Epsilon for the adjoint fixed point iterations. The default is 1e-5.
        is_adj_numerical_acceleration: bool
            Whether to use numerical acceleration in the adjoint state.
            The default is False.
        is_use_continuous_adj: bool
            Whether to use the adjoint state derived using the "continuous" way,
            aka the differentiate-then-discretize method. This option has been
            added to illustrate the paper TODO: add ref. The default is False.

        """
        self.grid: Geometry = grid
        self.time_params: TimeParameters = time_params
        if is_gravity:
            self.a_fl_model: AdjointFlowModel = DensityAdjointFlowModel(
                grid, time_params, is_use_continuous_adj
            )
        else:
            self.a_fl_model: AdjointFlowModel = SaturatedAdjointFlowModel(
                grid, time_params, is_use_continuous_adj
            )
        self.a_tr_model: AdjointTransportModel = AdjointTransportModel(
            grid, time_params, n_sp, afpi_eps, is_adj_numerical_acceleration
        )

    def clear_adjoint_sources(self) -> None:
        """
        Reset all adjoint sources to zero.
        """
        self.a_fl_model.clear_adjoint_sources()
        self.a_tr_model.clear_adjoint_sources()

    def init_adjoint_sources(
        self,
        fwd_model: ForwardModel,
        observables: Observables,
        hm_end_time: Optional[float] = None,
    ) -> None:
        """
        Initiate the adjoint variables.

        Parameters
        ----------
        model : ForwardModel
            The forward model.
        observables : Observables
            Sequence of observable instances for which to derive the adjoint sources.
        hm_end_time : Optional[float], optional
            Threshold time from which the observation are ignored, by default None.
        """

        # First set all to zero
        self.clear_adjoint_sources()

        if hm_end_time is not None:
            max_obs_time = min(fwd_model.time_params.time_elapsed, hm_end_time)
        else:
            max_obs_time = fwd_model.time_params.time_elapsed

        # get the number of observations used to scale the LS objective function
        # by the number of observables and obtain smaller gradients with smaller
        # objective functions
        n_obs = get_observables_values_as_1d_vector(
            observables, max_obs_time=max_obs_time
        ).size

        # get the adjoint variable for each observable
        for obs in object_or_object_sequence_to_list(observables):
            # adjoint sources for this observable to a sparse matrix

            array = {
                StateVariable.CONCENTRATION: self.a_tr_model.a_conc_sources[obs.sp],
                StateVariable.DENSITY: self.a_tr_model.a_density_sources,
                StateVariable.DIFFUSION: self.a_tr_model.a_diffusion_sources,
                StateVariable.DISPERSIVITY: self.a_tr_model.a_dispersivity_sources,
                StateVariable.HEAD: self.a_fl_model.a_head_sources,
                StateVariable.GRADE: self.a_tr_model.a_grade_sources[obs.sp],
                StateVariable.PERMEABILITY: self.a_fl_model.a_permeability_sources,
                StateVariable.POROSITY: self.a_tr_model.a_porosity_sources,
                StateVariable.PRESSURE: self.a_fl_model.a_pressure_sources,
                StateVariable.STORAGE_COEFFICIENT: (
                    self.a_fl_model.a_storage_coefficient_sources
                ),
            }[obs.state_variable]

            # Add the sparse array to the correct attribute
            res = csc_array(
                get_adjoint_sources_for_obs(
                    fwd_model, obs, n_obs, max_obs_time
                ).reshape(array.shape, order="F")
            )

            if obs.state_variable == StateVariable.CONCENTRATION:
                self.a_tr_model.a_conc_sources[obs.sp] += res
            elif obs.state_variable == StateVariable.DENSITY:
                self.a_tr_model.a_density_sources += res
            elif obs.state_variable == StateVariable.DIFFUSION:
                self.a_tr_model.a_diffusion_sources += res
            elif obs.state_variable == StateVariable.DISPERSIVITY:
                self.a_tr_model.a_dispersivity_sources += res
            elif obs.state_variable == StateVariable.HEAD:
                self.a_fl_model.a_head_sources += res
            elif obs.state_variable == StateVariable.GRADE:
                self.a_tr_model.a_grade_sources[obs.sp] += res
            elif obs.state_variable == StateVariable.PERMEABILITY:
                self.a_fl_model.a_permeability_sources += res
            elif obs.state_variable == StateVariable.POROSITY:
                self.a_tr_model.a_porosity_sources += res
            elif obs.state_variable == StateVariable.PRESSURE:
                self.a_fl_model.a_pressure_sources += res
            elif obs.state_variable == StateVariable.STORAGE_COEFFICIENT:
                self.a_fl_model.a_storage_coefficient_sources += res

    def reinit(self) -> None:
        """Reinitialize the adjoint state model. Set all arrays to zero."""
        self.a_fl_model.reinit()
        self.a_tr_model.reinit()
