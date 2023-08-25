"""Provide the models for the adjoint states."""
from __future__ import annotations

import numpy as np
from scipy.sparse import lil_matrix

from pyrtid.forward.models import (  # ConstantHead,; ZeroConcGradient,
    ForwardModel,
    Geometry,
    TimeParameters,
)
from pyrtid.inverse.obs import Observable, StateVariable
from pyrtid.utils.types import NDArrayFloat


class AdjointFlowModel:
    """Represent an adjoint flow model."""

    __slots__ = [
        "a_head",
        "a_u_darcy_x",
        "a_u_darcy_y",
        "a_head_sources",
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
        # Generally, not so many observation, so use a sparse matrix
        # instead of a dense array
        self.a_head_sources: lil_matrix = lil_matrix(
            (geometry.nx * geometry.ny, time_params.nt), dtype=np.float64
        )
        self.q_prev = lil_matrix(geometry.nx * geometry.ny)
        self.q_next = lil_matrix(geometry.nx * geometry.ny)


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
    q_prev_diffusion: lil_matrix
    q_next_diffusion: lil_matrix
    q_prev: lil_matrix
    q_next: lil_matrix
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

        self.a_conc_sources = np.zeros(
            (geometry.nx, geometry.ny, time_params.nt), dtype=np.float64
        )
        self.a_grade_sources = np.zeros(
            (geometry.nx, geometry.ny, time_params.nt), dtype=np.float64
        )

        # Adjoint source term from the adjoint geochem to the adjoint transport
        self.a_gch_src_term = np.zeros((geometry.nx, geometry.ny))

        self.q_prev_diffusion = lil_matrix(geometry.nx * geometry.ny)
        self.q_next_diffusion = lil_matrix(geometry.nx * geometry.ny)
        self.q_prev = lil_matrix(geometry.nx * geometry.ny)
        self.q_next = lil_matrix(geometry.nx * geometry.ny)
        self.afpi_eps = afpi_eps
        self.is_adj_numerical_acceleration = is_adj_numerical_acceleration


class AdjointModel:
    """Represent an adjoint model."""

    __slots__ = ["geometry", "time_params", "gch_params", "a_fl_model", "a_tr_model"]

    def __init__(
        self,
        geometry: Geometry,
        time_params: TimeParameters,
        afpi_eps: float,
        is_adj_numerical_acceleration: bool,
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

    @property
    def is_head_obs(self) -> bool:
        """Return whether there are head observations."""
        if self.a_fl_model.a_head_sources.count_nonzero() != 0:
            return True
        return False

    @property
    def is_mob_obs(self) -> bool:
        """Return whether there are mobile concentrations observations."""
        if np.any(self.a_tr_model.a_conc_sources):
            return True
        return False

    @property
    def is_immob_obs(self) -> bool:
        """Return whether there are mobile concentrations observations."""
        if np.any(self.a_tr_model.a_grade_sources):
            return True
        return False

    def set_adjoint_sources_from_obs(
        self, obs: Observable, model: ForwardModel
    ) -> None:
        """Set the adjoint sources to the correct model."""
        if obs.state_variable == StateVariable.CONCENTRATION:
            self.set_adjoint_sources_from_mob_obs(obs, model)
        elif obs.state_variable == StateVariable.HEAD:
            self.set_adjoint_sources_from_head_obs(obs, model)
        else:
            raise ValueError("Not a valid state variable type!")

    def set_adjoint_sources_from_mob_obs(
        self, obs: Observable, model: ForwardModel
    ) -> None:
        """Set the adjoint sources to the correct model."""
        try:
            # case obs.location is a numpy array
            self.a_tr_model.a_conc_sources[obs.location, obs.times] += (
                obs.values - model.tr_model.conc[obs.location, obs.times].ravel()
            ) / (obs.uncertainties**2)
        except IndexError:
            # case obs.location is a tuple of slices
            self.a_tr_model.a_conc_sources[(*obs.location, obs.times)] += (
                obs.values - model.tr_model.conc[(*obs.location, obs.times)].ravel()
            ) / (obs.uncertainties**2)

    def set_adjoint_sources_from_head_obs(
        self, obs: Observable, model: ForwardModel
    ) -> None:
        """Set the adjoint sources to the correct model."""
        try:
            # case obs.location is a numpy array
            self.a_fl_model.a_head_sources[obs.location, obs.times] += (
                obs.values - model.fl_model.head[obs.location, obs.times].ravel()
            ) / (obs.uncertainties**2)
        except IndexError:
            # case obs.location is a tuple of slices
            self.a_fl_model.a_head_sources[(*obs.location, obs.times)] += (
                obs.values - model.fl_model.head[(*obs.location, obs.times)].ravel()
            ) / (obs.uncertainties**2)
