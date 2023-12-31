"""Provide the models for the adjoint states."""
from __future__ import annotations

import numpy as np
from scipy.sparse import csc_matrix, lil_matrix

from pyrtid.forward.models import (  # ConstantHead,; ZeroConcGradient,
    ForwardModel,
    GeochemicalParameters,
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
        "a_sources",
        "q_prev",
        "q_next",
    ]

    def __init__(self, geometry: Geometry, time_params: TimeParameters) -> None:
        """Initialize the instance."""
        self.a_head = np.zeros(
            (geometry.nx, geometry.ny, time_params.nt + 1), dtype=np.float64
        )
        self.a_u_darcy_x = np.zeros(
            (geometry.nx - 1, geometry.ny, time_params.nt + 1), dtype=np.float64
        )
        self.a_u_darcy_y = np.zeros(
            (geometry.nx, geometry.ny - 1, time_params.nt + 1), dtype=np.float64
        )
        # TODO: see if we can use sparse matrices for that
        self.a_sources = np.zeros(
            (geometry.nx, geometry.ny, time_params.nt + 1), dtype=np.float64
        )
        self.q_prev = lil_matrix(geometry.nx * geometry.ny)
        self.q_next = lil_matrix(geometry.nx * geometry.ny)


class AdjointTransportModel:
    """Represent an adjoint flow model."""

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
    ]

    def __init__(
        self,
        geometry: Geometry,
        time_params: TimeParameters,
        gch_params: GeochemicalParameters,
    ) -> None:
        """Initialize the instance."""
        self.a_conc: NDArrayFloat = np.zeros(
            (geometry.nx, geometry.ny, time_params.nt + 1), dtype=np.float64
        )
        self.a_conc_prev: NDArrayFloat = np.zeros(
            (geometry.nx, geometry.ny), dtype=np.float64
        )
        self.a_grade: NDArrayFloat = np.zeros(
            (geometry.nx, geometry.ny, time_params.nt + 1), dtype=np.float64
        )

        self.a_conc_sources = np.zeros(
            (geometry.nx, geometry.ny, time_params.nt + 1), dtype=np.float64
        )
        self.a_grade_sources = np.zeros(
            (geometry.nx, geometry.ny, time_params.nt + 1), dtype=np.float64
        )

        # Adjoint source term from the adjoint geochem to the adjoint transport
        self.a_gch_src_term = np.zeros((geometry.nx, geometry.ny))

        self.q_prev_diffusion = lil_matrix(geometry.nx * geometry.ny)
        self.q_next_diffusion = lil_matrix(geometry.nx * geometry.ny)
        self.q_prev = csc_matrix(geometry.nx * geometry.ny)
        self.q_next = csc_matrix(geometry.nx * geometry.ny)


class AdjointModel:
    """Represent an adjoint model."""

    __slots__ = ["geometry", "time_params", "gch_params", "a_fl_model", "a_tr_model"]

    def __init__(
        self,
        geometry: Geometry,
        time_params: TimeParameters,
        gch_params: GeochemicalParameters,
    ) -> None:
        """
        Initialize the instance.

        Parameters
        ----------
        geometry : Geometry
            _description_
        time_params : TimeParameters
            _description_
        gch_params : GeochemicalParameters
            _description_
        """
        self.geometry: Geometry = geometry
        self.time_params: TimeParameters = time_params
        self.gch_params: GeochemicalParameters = gch_params
        self.a_fl_model: AdjointFlowModel = AdjointFlowModel(geometry, time_params)
        self.a_tr_model: AdjointTransportModel = AdjointTransportModel(
            geometry, time_params, gch_params
        )

    @property
    def is_head_obs(self) -> bool:
        """Return whether there are head observations."""
        if np.any(self.a_fl_model.a_head_sources):
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
            self.a_tr_model.a_sources[obs.location, obs.timesteps] += (
                obs.values - model.tr_model.conc[obs.location, obs.timesteps].ravel()
            ) / (obs.uncertainties**2)
        except IndexError:
            # case obs.location is a tuple of slices
            self.a_tr_model.a_sources[(*obs.location, obs.timesteps)] += (
                obs.values - model.tr_model.conc[(*obs.location, obs.timesteps)].ravel()
            ) / (obs.uncertainties**2)

    def set_adjoint_sources_from_head_obs(
        self, obs: Observable, model: ForwardModel
    ) -> None:
        """Set the adjoint sources to the correct model."""
        try:
            # case obs.location is a numpy array
            self.a_fl_model.a_sources[obs.location, obs.timesteps] += (
                obs.values - model.fl_model.head[obs.location, obs.timesteps].ravel()
            ) / (obs.uncertainties**2)
        except IndexError:
            # case obs.location is a tuple of slices
            self.a_fl_model.a_sources[(*obs.location, obs.timesteps)] += (
                obs.values - model.fl_model.head[(*obs.location, obs.timesteps)].ravel()
            ) / (obs.uncertainties**2)
