"""Provide a representation of observables."""
import json
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import numpy as np

from pyrtid.forward import ForwardModel
from pyrtid.utils.enum import StrEnum
from pyrtid.utils.types import (
    NDArrayFloat,
    NDArrayInt,
    object_or_object_sequence_to_list,
)


class StateVariable(StrEnum):
    """Type of observable existing."""

    DIFFUSION = "diffusion"
    POROSITY = "porosity"
    PERMEABILITY = "permeability"
    HEAD = "head"
    CONCENTRATION = "concentration"
    MINERAL_GRADE = "grade"


@dataclass
class Observable:
    """
    Class representing observations data within time at a defined location.

    Attributes
    ----------
    state_variable: StateVariable
        Name of the state variable or the parameter being observed.
    location: slice
        Location of the observation in the grid.
    timesteps: NDArrayInt
        Timesteps matching the values.
    values: NDArrayFloat
        Observed values.
    uncertainties: Union[float, NDArrayFloat]
        Absolute uncertainties associated with the observed values.
    """

    state_variable: StateVariable
    location: Union[NDArrayInt, Tuple[slice, slice]]
    timesteps: NDArrayInt
    values: NDArrayFloat
    uncertainties: Union[float, NDArrayFloat] = np.array([])

    def __post_init__(self) -> None:
        """Construct the class."""
        self.uncertainties = np.array(self.uncertainties)
        if self.uncertainties.size == 0:
            self.uncertainties = np.ones(self.values.shape)
        elif self.uncertainties.size == 1:
            self.uncertainties = (
                np.ones(self.values.shape) * self.uncertainties.ravel()[0]
            )
        if self.uncertainties.size != self.values.size:
            raise ValueError(
                "Uncertainties should be a float values or a numpy "
                "array with the same dimension as values."
            )
        if self.timesteps.size != self.values.size:
            raise ValueError(
                "Timesteps should be a float values or a numpy "
                "array with the same dimension as values."
            )

        if isinstance(self.location, np.ndarray):
            if self.location.dim != 2:
                raise ValueError(
                    "Location should be a 2D np.array or a tuple of two slices !"
                )

    def __str__(self):
        """Represent the class object as a string."""
        return json.dumps(
            self.__dict__, indent=0, sort_keys=False, default=str
        ).replace("null", "None")


def get_observable_values(
    obs: Observable, hm_end_time: Optional[float] = None
) -> NDArrayFloat:
    return obs.values


def get_observable_uncertainties(
    obs: Observable, hm_end_time: Optional[float] = None
) -> NDArrayFloat:
    return obs.uncertainties


def get_predictions_matching_observations(
    model: ForwardModel, observables: Union[Observable, Sequence[Observable]]
) -> NDArrayFloat:
    """Return the vector of predictions matching the observations."""
    res = []
    for obs in object_or_object_sequence_to_list(observables):
        does_support_time = True
        if obs.state_variable == StateVariable.CONCENTRATION:
            array: NDArrayFloat = model.tr_model.conc
        elif obs.state_variable == StateVariable.HEAD:
            array = model.fl_model.head
        elif obs.state_variable == StateVariable.MINERAL_GRADE:
            array = model.tr_model.grade
        elif obs.state_variable == StateVariable.PERMEABILITY:
            array = model.fl_model.permeability
            does_support_time = False
        elif obs.state_variable == StateVariable.POROSITY:
            array = model.tr_model.porosity
            does_support_time = False
        else:
            raise ValueError("Not a valid state variable or parameter type!")

        if does_support_time:
            try:
                # case obs.location is a numpy array
                res.append(array[np.array(obs.location), obs.timesteps].ravel())
            except IndexError:
                # case obs.location is a tuple of slices
                res.append(array[(*obs.location, obs.timesteps)].ravel())
        else:  # state variable constant within time
            try:
                # case obs.location is a numpy array
                res.append(array[obs.location].ravel())
            except IndexError:
                # case obs.location is a tuple of slices
                res.append(array[(*obs.location,)].ravel())

    return np.stack(res).ravel()


def get_observables_values_as_1d_vector(
    observables: Union[Observable, Sequence[Observable]],
    hm_end_time: Optional[float] = None,
) -> NDArrayFloat:
    """
    Return the values of all given observables as a 1D vector.

    Order is preserved.
    """
    return np.stack(
        [
            get_observable_values(obs, hm_end_time)
            for obs in object_or_object_sequence_to_list(observables)
        ]
    ).ravel()


def get_observables_uncertainties_as_1d_vector(
    observables: Union[Observable, Sequence[Observable]],
    hm_end_time: Optional[float] = None,
) -> NDArrayFloat:
    """Return the uncertainties of all observables as a 1D vector."""
    return np.stack(
        [
            get_observable_uncertainties(obs, hm_end_time)
            for obs in object_or_object_sequence_to_list(observables)
        ]
    ).ravel()
