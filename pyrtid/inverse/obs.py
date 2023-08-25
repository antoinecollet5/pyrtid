"""Provide a representation of observables."""
import json
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from pyrtid.forward import ForwardModel
from pyrtid.utils import node_number_to_indices
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
    PRESSURE = "pressure"
    CONCENTRATION = "concentration"
    MINERAL_GRADE = "grade"
    DENSITY = "density"


class Observable:
    """
    Class representing observations data within time at a defined location.

    Note
    ----
    It is fine if the times are not sorted in increasing order.

    Attributes
    ----------
    state_variable: StateVariable
        Name of the state variable or the parameter being observed.
    node_indices: NDArrayInt
        Node indices to locate of the observation in the grid.
    times: NDArrayInt
        Times matching the values.
    values: NDArrayFloat
        Observed values.
    uncertainties: NDArrayFloat
        Absolute uncertainties associated with the observed values.
    """

    __slots__ = ["state_variable", "node_indices", "times", "values", "uncertainties"]

    def __init__(
        self,
        state_variable: StateVariable,
        node_indices: Union[NDArrayInt, List[int], int],
        times: NDArrayFloat,
        values: NDArrayFloat,
        uncertainties: Optional[Union[float, NDArrayFloat]] = None,
    ) -> None:
        """_summary_

        Parameters
        ----------
        state_variable: StateVariable
            Name of the state variable or the parameter being observed.
        node_indices: Union[NDArrayInt, List[int], int]
            Location of the observation in the grid (indices from 0 to nx * ny - 1).
        timesteps: NDArrayFloat
            Timesteps matching the values.
        values: NDArrayFloat
            Observed values.
        uncertainties: Optional[Union[float, NDArrayFloat]]
            Absolute uncertainties associated with the observed values.

        Raises
        ------
        ValueError
            _description_
        ValueError
            _description_
        ValueError
            _description_
        """

        self.state_variable = state_variable
        self.node_indices = np.sort(np.array(node_indices).ravel())
        self.times = times
        self.values = values

        _uncertainties = (
            np.array(uncertainties) if uncertainties is not None else np.array([])
        )

        if _uncertainties.size == 0:
            self.uncertainties = np.ones(self.values.shape)
        elif _uncertainties.size == 1:
            self.uncertainties = np.ones(self.values.shape) * _uncertainties.ravel()[0]
        elif _uncertainties.size != self.values.size:
            raise ValueError(
                "Uncertainties should be a float values or a numpy "
                "array with the same dimension as values."
            )
        else:
            self.uncertainties = _uncertainties

        if self.times.size != self.values.size:
            raise ValueError(
                "Timesteps should be a float values or a numpy "
                "array with the same dimension as values."
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


def get_array_from_state_variable(
    model: ForwardModel, state_variable: StateVariable
) -> NDArrayFloat:
    if state_variable == StateVariable.CONCENTRATION:
        return model.tr_model.conc
    if state_variable == StateVariable.HEAD:
        return model.fl_model.head
    if state_variable == StateVariable.PRESSURE:
        return model.fl_model.pressure
    if state_variable == StateVariable.MINERAL_GRADE:
        return model.tr_model.grade
    if state_variable == StateVariable.PERMEABILITY:
        return model.fl_model.permeability
    if state_variable == StateVariable.POROSITY:
        return model.tr_model.porosity
    if state_variable == StateVariable.DIFFUSION:
        return model.tr_model.diffusion
    if state_variable == StateVariable.DENSITY:
        return model.tr_model.density
    else:
        raise ValueError(
            f'"{state_variable}" is not a valid state variable or parameter type!'
        )


def get_predictions_matching_observations(
    model: ForwardModel, observables: Union[Observable, Sequence[Observable]]
) -> NDArrayFloat:
    """Return the vector of predictions matching the observations."""
    res = []
    for obs in object_or_object_sequence_to_list(observables):
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


def get_times_idx_before_after_obs(
    obs_times: NDArrayFloat, simu_times: NDArrayFloat
) -> Tuple[NDArrayInt, NDArrayInt]:
    """
    Get the calcuated times before and after the observation times.

    Parameters
    ----------
    obs_times : NDArrayFloat
        Times of observated values.
    simu_times : NDArrayFloat
        Times of simulated values.

    Returns
    -------
    Tuple[NDArrayFloat, NDArrayFloat]
        Indices of simulated values coming after and before the observed values.
    """
    simu_times_before_idx = []
    simu_times_after_idx = []

    idx = 0
    for obs_time in obs_times:
        while obs_time > simu_times[idx]:
            idx += 1

        simu_times_before_idx.append(idx - 1)
        simu_times_after_idx.append(idx)

    return np.array(simu_times_before_idx), np.array(simu_times_after_idx)


def get_weights(
    obs_times: NDArrayFloat,
    simu_times: NDArrayFloat,
    before_idx: NDArrayInt,
    after_idx: NDArrayInt,
) -> Tuple[NDArrayFloat, NDArrayFloat]:
    """
    Get the weights to apply on the calculated values after and before observations.

    Parameters
    ----------
    obs_times : NDArrayFloat
        Times of observated values.
    simu_times : NDArrayFloat
        Times of simulated values.
    before_idx : NDArrayInt
        Indices of simulated values coming before the observed values.
    after_idx : NDArrayInt
        Indices of simulated values coming after the observed values.

    Returns
    -------
    Tuple[NDArrayFloat, NDArrayFloat]
        Weights to apply on simulated values before and after observations.
    """
    den = simu_times[after_idx] - simu_times[before_idx]
    weights_before = 1 - (obs_times - simu_times[before_idx]) / den
    weights_before[before_idx < 0] = 0.0  # Handle obs at time zero.
    weights_after = 1 - (simu_times[after_idx] - obs_times) / den

    return weights_before, weights_after


def get_values_matching_node_indices(
    node_indices: NDArrayInt, input_values: NDArrayFloat
) -> NDArrayFloat:
    """
    Return the values for the given node_indices with shape.

    The output shape is (|node_indices|, [nt]), where |.| means cardinality.

    Parameters
    ----------
    node_indices : NDArrayInt
        Indices in the grid from 0 to nx * ny -1.
    input_values : NDArrayFloat
        Array of input values with shape (nx, ny, nt) or (nx, ny).

    Returns
    -------
    NDArrayFloat
        Simulated values at the observation location
    """
    nx, ny = input_values.shape[:2]
    X, Y, _ = node_number_to_indices(node_indices, nx=nx, ny=ny)
    if len(input_values.shape) == 3:
        return input_values[X, Y, :]
    # state variable constant within time
    return input_values[X, Y]
