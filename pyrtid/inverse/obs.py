"""Provide a representation of observables."""
import json
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from pyrtid.forward import ForwardModel
from pyrtid.utils import node_number_to_indices
from pyrtid.utils.enum import StrEnum
from pyrtid.utils.means import MeanType, get_mean_values_for_last_axis
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
    times: NDArrayFloat
        Times matching the values.
    values: NDArrayFloat
        Observed values.
    uncertainties: NDArrayFloat
        Absolute uncertainties associated with the observed values.
    mean_type: MeanType
        Type of mean used to averaged the simulated values when the observations
        are performed over several meshes.
    """

    __slots__ = [
        "state_variable",
        "node_indices",
        "times",
        "values",
        "uncertainties",
        "_mean_type",
    ]

    def __init__(
        self,
        state_variable: StateVariable,
        node_indices: Union[NDArrayInt, List[int], int],
        times: NDArrayFloat,
        values: NDArrayFloat,
        uncertainties: Optional[Union[float, NDArrayFloat]] = None,
        mean_type: Optional[MeanType] = None,
    ) -> None:
        """_summary_

        Parameters
        ----------
        state_variable: StateVariable
            Name of the state variable or the parameter being observed.
        node_indices: Union[NDArrayInt, List[int], int]
            Location of the observation in the grid (indices from 0 to nx * ny - 1).
        times: NDArrayFloat
            Timesteps matching the values.
        values: NDArrayFloat
            Observed values.
        uncertainties: Optional[Union[float, NDArrayFloat]]
            Absolute uncertainties associated with the observed values.
        mean_type: Optional[MeanType] = None
            Type of mean used to averaged the simulated values when the observations
            are performed over several meshes. If None, an harmonic mean is used
            for diffusion and hydraulic conductivity coefficients, otherwise,
            an arithmetic mean is used.
            The default is None.
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
                "``uncertainties`` parameter should be a float value or a numpy "
                "array with the same dimension as the ``values`` parameter."
            )
        else:
            self.uncertainties = _uncertainties

        if self.times.size != self.values.size:
            raise ValueError(
                "``times`` parameter should be a float value or a numpy "
                "array with the same dimension as the ``values`` parameter."
            )

        self.mean_type = mean_type

    @property
    def mean_type(self) -> MeanType:
        """Return the mean type used to interpolate values over several meshes."""
        return self._mean_type

    @mean_type.setter
    def mean_type(self, value: Optional[MeanType]) -> None:
        """Set the mean type used to interpolate values over several meshes."""
        if value is None:
            if self.state_variable in [
                StateVariable.DIFFUSION,
                StateVariable.PERMEABILITY,
            ]:
                self._mean_type = MeanType.HARMONIC
            else:
                self._mean_type = MeanType.ARITHMETIC
        else:
            self._mean_type = value

    def __str__(self) -> str:
        """Return a string representation of the instance."""
        return json.dumps(
            {
                "state_variable": self.state_variable,
                "node_indices": self.node_indices,
                "nb_values": self.values.size,
                "min_value": np.min(self.values),
                "max_value": np.max(self.values),
                "min_time": np.min(self.times),
                "max_time": np.max(self.times),
                "min_std": np.max(self.uncertainties),
                "mean_type": self.mean_type,
            },
            indent=4,
            sort_keys=False,
            default=str,
        ).replace("null", "None")


def _get_obs_ascending_time_sorting_permutations(
    obs: Observable, max_time: Optional[float] = None
) -> NDArrayInt:
    """
    Get the permutations required to sort the observation time in ascending order.

    Parameters
    ----------
    obs : Observable
        The observble instance.
    max_time : Optional[float], optional
        Maximum time value to consider, by default None

    Returns
    -------
    NDArrayInt
        The permutations as an array of int giving the new absolute positions.
    """
    valid_indices = np.arange(obs.times.size)
    if max_time is not None:
        valid_indices = valid_indices[obs.times <= max_time]
    sorted_indices = obs.times.argsort()
    return sorted_indices[np.isin(sorted_indices, valid_indices)]


def get_sorted_observable_times(
    obs: Observable, max_time: Optional[float] = None
) -> NDArrayFloat:
    """
    Get the observation times sorted in ascending order.

    Parameters
    ----------
    obs : Observable
        The observable instance.
    max_time : Optional[float], optional
        Maximum time value to consider, by default None

    Returns
    -------
    NDArrayFloat
        Observation times sorted in ascending order.
    """
    return obs.times[
        _get_obs_ascending_time_sorting_permutations(obs, max_time)
    ].flatten()


def get_sorted_observable_values(
    obs: Observable, max_time: Optional[float] = None
) -> NDArrayFloat:
    """
    Get the observation values sorted by ascending corresponding times.

    Parameters
    ----------
    obs : Observable
        The observable instance.
    max_time : Optional[float], optional
        Maximum time value to consider, by default None

    Returns
    -------
    NDArrayFloat
        Observation values sorted by ascending corresponding times.
    """
    return obs.values[
        _get_obs_ascending_time_sorting_permutations(obs, max_time)
    ].flatten()


def get_sorted_observable_uncertainties(
    obs: Observable, max_time: Optional[float] = None
) -> NDArrayFloat:
    """
    Get the observation uncertainties sorted by ascending corresponding times.

    Parameters
    ----------
    obs : Observable
        The observable instance.
    max_time : Optional[float], optional
        Maximum time value to consider, by default None

    Returns
    -------
    NDArrayFloat
        Observation uncertainties sorted by ascending corresponding times.
    """
    return obs.uncertainties[
        _get_obs_ascending_time_sorting_permutations(obs, max_time)
    ].flatten()


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


def get_observables_values_as_1d_vector(
    observables: Union[Observable, Sequence[Observable]],
    max_obs_time: Optional[float] = None,
) -> NDArrayFloat:
    """
    Return the values of all given observables as a 1D vector.

    Note
    ----
    The observation values are sorted first by Observable instance (observation
    location) and by ascending time at a second level.
    """
    return np.hstack(
        [
            get_sorted_observable_values(obs, max_obs_time)
            for obs in object_or_object_sequence_to_list(observables)
        ]
    ).ravel()


def get_observables_uncertainties_as_1d_vector(
    observables: Union[Observable, Sequence[Observable]],
    max_obs_time: Optional[float] = None,
) -> NDArrayFloat:
    """Return the uncertainties of all observables as a 1D vector.

    Note
    ----
    The observation values are sorted first by Observable instance (observation
    location) and by ascending time at a second level.
    """
    return np.hstack(
        [
            get_sorted_observable_uncertainties(obs, max_obs_time)
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


def get_interp_simu_values_matching_obs_times(
    obs_times: NDArrayFloat,
    simu_times: NDArrayFloat,
    simu_values: NDArrayFloat,
) -> NDArrayFloat:
    """
    Get an array of interpolated values that match the obervation times.

    TODO: add ref to paper + a scheme.

    A simple linear interpolation is performed with two values only (the one before
    and the one after the observation time.)

    Note
    ----
    If measured on several nodes, the simulated values should already
    have been averaged at this point.

    Parameters
    ----------
    obs_times : NDArrayFloat
        1D array of observation times.
    simu_times : NDArrayFloat
        1D array of simulation times.
    simu_values : NDArrayFloat
        1D array of simulated values.

    Returns
    -------
    NDArrayFloat
        Simulated values interpolated at observation times.

    """
    idx_before, idx_after = get_times_idx_before_after_obs(obs_times, simu_times)
    weights_before, weights_after = get_weights(
        obs_times, simu_times, idx_before, idx_after
    )
    return (
        weights_before * simu_values[idx_before]
        + weights_after * simu_values[idx_after]
    )


def get_simulated_values_matching_obs(
    model: ForwardModel,
    obs: Observable,
    max_obs_time: Optional[float] = None,
) -> NDArrayFloat:
    """
    Get the simulated values matching the given observable and the hm end time.

    Parameters
    ----------
    model : ForwardModel
        The forward model.
    obs : Observable
        The observable instance.
    ldt : List[float]
        The list of timesteps used to solve the forward model. The simulation times
        are derived from the list.
    max_obs_time : Optional[float], optional
        Maximum time for which to consider an obervation value, by default None.

    Returns
    -------
    NDArrayFloat
        Simulated values matching the given observable and the hm end time.
    """
    simu_times = np.cumsum([0] + model.time_params.ldt)
    if max_obs_time is not None:
        max_obs_time = min(np.max(simu_times), max_obs_time)
    else:
        max_obs_time = np.max(simu_times)
    obs_times = get_sorted_observable_times(obs, max_obs_time)
    _simu_values = get_mean_values_for_last_axis(
        get_values_matching_node_indices(
            obs.node_indices, get_array_from_state_variable(model, obs.state_variable)
        ),
        mean_type=obs.mean_type,
        weights=None,
    )
    return get_interp_simu_values_matching_obs_times(
        obs_times, simu_times, _simu_values
    )


def get_predictions_matching_observations(
    model: ForwardModel,
    observables: Union[Observable, Sequence[Observable]],
    max_obs_time: Optional[float] = None,
) -> NDArrayFloat:
    """
    Return the 1D vector of predictions matching the observations.

    Parameters
    ----------
    model : ForwardModel
        _description_
    observables : Union[Observable, Sequence[Observable]]
        _description_
    max_obs_time : Optional[float], optional
        Maximum time for which to consider an obervation value, by default None.

    Returns
    -------
    NDArrayFloat
        _description_
    """
    res = []
    for obs in object_or_object_sequence_to_list(observables):
        res.append(
            get_simulated_values_matching_obs(model, obs, max_obs_time).flatten()
        )
    return np.hstack(res).ravel()
