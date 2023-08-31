"""Provide the models for the adjoint states."""
from __future__ import annotations

from typing import Optional

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


class AdjointFlowModel:
    """Represent an adjoint flow model."""

    __slots__ = [
        "a_head",
        "a_u_darcy_x",
        "a_u_darcy_y",
        "a_head_sources",
        "a_pressure_sources",
        "a_density_sources",
        "a_permeability_sources",
        "q_prev",
        "q_next",
    ]

    def __init__(self, geometry: Geometry, time_params: TimeParameters) -> None:
        """Initialize the instance."""
        self.a_head = np.zeros(
            (geometry.nx, geometry.ny, time_params.nt), dtype=np.float64
        )
        self.a_u_darcy_x = np.zeros(
            (geometry.nx - 1, geometry.ny, time_params.nt), dtype=np.float64
        )
        self.a_u_darcy_y = np.zeros(
            (geometry.nx, geometry.ny - 1, time_params.nt), dtype=np.float64
        )
        # Generally, not so many observations, so only a few adjoint variable,
        # so use a sparse matrix instead of a dense array
        # NOTE: rows are meshes, and columns are time indices
        # We use csc format for fast column (time) slicing
        self.a_head_sources: csc_array = csc_array(
            (geometry.nx * geometry.ny, time_params.nt), dtype=np.float64
        )
        self.a_pressure_sources: csc_array = csc_array(
            (geometry.nx * geometry.ny, time_params.nt), dtype=np.float64
        )
        self.a_density_sources: csc_array = csc_array(
            (geometry.nx * geometry.ny, time_params.nt), dtype=np.float64
        )
        # does not vary in time
        self.a_permeability_sources: csc_array = csc_array(
            (geometry.nx * geometry.ny, 1), dtype=np.float64
        )

        self.q_prev = lil_array(geometry.nx * geometry.ny)
        self.q_next = lil_array(geometry.nx * geometry.ny)

    def clear_adjoint_sources(self) -> None:
        """
        Reset all adjoint sources to zero.
        """
        self.a_head_sources = csc_array(self.a_head_sources.shape)
        self.a_pressure_sources = csc_array(self.a_pressure_sources.shape)
        self.a_density_sources = csc_array(self.a_density_sources.shape)
        self.a_permeability_sources = csc_array(self.a_permeability_sources.shape)


class AdjointTransportModel:
    """
    Represent an adjoint flow model.

    Attributes
    ----------
    a_conc: NDArrayFloat
    a_conc_prev: NDArrayFloat
    a_grade: NDArrayFloat
    a_conc_sources: NDArrayFloat
    a_grade_sources: NDArrayFloat
    q_prev_diffusion: lil_array
    q_next_diffusion: lil_array
    q_prev: lil_array
    q_next: lil_array
    a_gch_src_term: NDArrayFloat
    afpi_eps: float
    is_adj_numerical_acceleration: bool
    """

    __slots__ = [
        "a_conc",
        "a_conc_prev",
        "a_grade",
        "a_conc_sources",
        "a_grade_sources",
        "a_porosity_sources",
        "a_diffusion_sources",
        "a_density_sources",
        "q_prev_diffusion",
        "q_next_diffusion",
        "q_prev",
        "q_next",
        "a_gch_src_term",
        "afpi_eps",
        "is_adj_numerical_acceleration",
    ]

    def __init__(
        self,
        geometry: Geometry,
        time_params: TimeParameters,
        afpi_eps: float = 1e-5,
        is_adj_numerical_acceleration: bool = False,
    ) -> None:
        """
        Initialize the instance.

        Parameters
        ----------
        geometry: Geometry
            Simulation geometry definition.
        time_params: TimeParameters
            Simulation time parameters (duration, timesteps, etc.)
        afpi_eps: float

        is_numerical_acceleration: bool

        """
        self.a_conc: NDArrayFloat = np.zeros(
            (geometry.nx, geometry.ny, time_params.nt), dtype=np.float64
        )
        self.a_conc_prev: NDArrayFloat = np.zeros(
            (geometry.nx, geometry.ny), dtype=np.float64
        )
        self.a_grade: NDArrayFloat = np.zeros(
            (geometry.nx, geometry.ny, time_params.nt), dtype=np.float64
        )

        # Generally, not so many observations, so only a few adjoint variable,
        # so use a sparse matrix instead of a dense array
        # NOTE: rows are meshes, and columns are time indices
        # We use csc format for fast column (time) slicing
        self.a_conc_sources = csc_array(
            (geometry.nx * geometry.ny, time_params.nt), dtype=np.float64
        )
        self.a_grade_sources = csc_array(
            (geometry.nx * geometry.ny, time_params.nt), dtype=np.float64
        )
        self.a_porosity_sources = csc_array(
            (geometry.nx * geometry.ny, 1), dtype=np.float64
        )
        self.a_diffusion_sources = csc_array(
            (geometry.nx * geometry.ny, 1), dtype=np.float64
        )

        # Adjoint source term from the adjoint geochem to the adjoint transport
        self.a_gch_src_term = np.zeros((geometry.nx, geometry.ny), dtype=np.float64)

        self.q_prev_diffusion = lil_array(geometry.nx * geometry.ny)
        self.q_next_diffusion = lil_array(geometry.nx * geometry.ny)
        self.q_prev = lil_array(geometry.nx * geometry.ny)
        self.q_next = lil_array(geometry.nx * geometry.ny)
        self.afpi_eps = afpi_eps
        self.is_adj_numerical_acceleration = is_adj_numerical_acceleration

    def clear_adjoint_sources(self) -> None:
        """Reset all adjoint sources to zero."""
        self.a_conc_sources = csc_array(self.a_conc_sources.shape)
        self.a_grade_sources = csc_array(self.a_grade_sources.shape)
        self.a_porosity_sources = csc_array(self.a_porosity_sources.shape)
        self.a_diffusion_sources = csc_array(self.a_diffusion_sources.shape)


class AdjointModel:
    """Represent an adjoint model."""

    __slots__ = ["geometry", "time_params", "gch_params", "a_fl_model", "a_tr_model"]

    def __init__(
        self,
        geometry: Geometry,
        time_params: TimeParameters,
        afpi_eps: float = 1e-5,
        is_adj_numerical_acceleration: bool = False,
    ) -> None:
        """
        Initialize the instance.

        Parameters
        ----------
        geometry: Geometry
            Simulation geometry definition.
        time_params: TimeParameters
            Simulation time parameters (duration, timesteps, etc.)
        afpi_eps: float
        is_numerical_acceleration: bool
        """
        self.geometry: Geometry = geometry
        self.time_params: TimeParameters = time_params
        self.a_fl_model: AdjointFlowModel = AdjointFlowModel(geometry, time_params)
        self.a_tr_model: AdjointTransportModel = AdjointTransportModel(
            geometry, time_params, afpi_eps, is_adj_numerical_acceleration
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

        # get the number of observations used to scale the objective function
        n_obs = get_observables_values_as_1d_vector(
            observables, max_obs_time=max_obs_time
        ).size

        # get the adjoint variable for each observable
        for obs in object_or_object_sequence_to_list(observables):
            # adjoint sources for this observable to a sparse matrix

            array = {
                StateVariable.CONCENTRATION: self.a_tr_model.a_conc_sources,
                StateVariable.DENSITY: self.a_fl_model.a_density_sources,
                StateVariable.DIFFUSION: self.a_tr_model.a_diffusion_sources,
                StateVariable.HEAD: self.a_fl_model.a_head_sources,
                StateVariable.GRADE: self.a_tr_model.a_grade_sources,
                StateVariable.PERMEABILITY: self.a_fl_model.a_permeability_sources,
                StateVariable.POROSITY: self.a_tr_model.a_porosity_sources,
                StateVariable.PRESSURE: self.a_fl_model.a_pressure_sources,
            }[obs.state_variable]

            # Add the sparse array to the correct attribute
            res = csc_array(
                get_adjoint_sources_for_obs(
                    fwd_model, obs, n_obs, max_obs_time
                ).reshape(array.shape, order="F")
            )

            if obs.state_variable == StateVariable.CONCENTRATION:
                self.a_tr_model.a_conc_sources += res
            elif obs.state_variable == StateVariable.DENSITY:
                self.a_fl_model.a_density_sources += res
            elif obs.state_variable == StateVariable.DIFFUSION:
                self.a_tr_model.a_diffusion_sources += res
            elif obs.state_variable == StateVariable.HEAD:
                self.a_fl_model.a_head_sources += res
            elif obs.state_variable == StateVariable.GRADE:
                self.a_tr_model.a_grade_sources += res
            elif obs.state_variable == StateVariable.PERMEABILITY:
                self.a_fl_model.a_permeability_sources += res
            elif obs.state_variable == StateVariable.POROSITY:
                self.a_tr_model.a_porosity_sources += res
            elif obs.state_variable == StateVariable.PRESSURE:
                self.a_fl_model.a_pressure_sources += res
