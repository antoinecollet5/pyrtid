"""Provide a representation of observables."""

from __future__ import annotations

import json
from typing import Optional, Sequence, Tuple, Union

import numpy as np

from pyrtid.forward import ForwardModel
from pyrtid.utils import node_number_to_indices
from pyrtid.utils.enum import StrEnum
from pyrtid.utils.means import (
    MeanType,
    get_mean_values_for_last_axis,
    get_mean_values_gradient_for_last_axis,
)
from pyrtid.utils.types import (
    Int,
    NDArrayFloat,
    NDArrayInt,
    object_or_object_sequence_to_list,
)


class StateVariable(StrEnum):
    """Type of observable existing."""

    CONCENTRATION = "concentration"
    DENSITY = "density"
    DIFFUSION = "diffusion"
    HEAD = "head"
    GRADE = "grade"
    PERMEABILITY = "permeability"
    POROSITY = "porosity"
    PRESSURE = "pressure"
    STORAGE_COEFFICIENT = "storage_coefficient"


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
        are performed over several grid cells.
    perturbations: NDArrayFloat
        Perturbations to add to the values when performing the inversion. This
        is only useful for the ensemble method when the observed values
        are perturbed with samples from N(0, R).
    sp: Optional[int]
        Index of the concentration being observed. Must be provided if a
        concentration or a grade is observed.
    """

    __slots__ = [
        "state_variable",
        "node_indices",
        "times",
        "_values",
        "uncertainties",
        "_mean_type",
        "perturbations",
        "sp",
    ]

    def __init__(
        self,
        state_variable: StateVariable,
        node_indices: Int,
        times: NDArrayFloat,
        values: NDArrayFloat,
        uncertainties: Optional[Union[float, NDArrayFloat]] = None,
        mean_type: Optional[MeanType] = None,
        sp: Optional[int] = None,
    ) -> None:
        """
        Initiate the instance.

        Warning
        -------
        For the control parameters such as porosity or permeability which are constant
        in time, we consider that all observations are done at time 0, no matter
        what is set by the user.

        Parameters
        ----------
        state_variable: StateVariable
            Name of the state variable or the parameter being observed.
        node_indices: Int
            Location of the observation in the grid (indices from 0 to nx * ny - 1).
        times: NDArrayFloat
            Timesteps matching the values.
        values: NDArrayFloat
            Observed values.
        uncertainties: Optional[Union[float, NDArrayFloat]]
            Absolute uncertainties associated with the observed values.
        mean_type: Optional[MeanType] = None
            Type of mean used to averaged the simulated values when the observations
            are performed over several grid cells. If None, an harmonic mean is used
            for diffusion and hydraulic conductivity coefficients, otherwise,
            an arithmetic mean is used.
            The default is None.
        """

        self.state_variable = state_variable
        self.node_indices = np.sort(np.array(node_indices).ravel())
        self.times = times.ravel()
        self.set_values(values.ravel())
        # perturbations must be initialized before any ".values" attribute call
        self.perturbations = np.zeros(values.shape)

        _uncertainties = (
            np.array(uncertainties) if uncertainties is not None else np.array([])
        ).ravel()

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

        if (
            self.state_variable in [StateVariable.CONCENTRATION, StateVariable.GRADE]
            and sp is None
        ):
            raise ValueError(
                "sp must be provided when observing grades or concentrations!"
            )

        # To avoid having None values for typing
        if sp is not None:
            self.sp: int = sp
        else:
            self.sp = 0

    @property
    def values(self) -> NDArrayFloat:
        """
        Return the values plus the optional perturbations.

        This is read-only. Use the method set_values to update the observations.
        """
        return self._values + self.perturbations

    def set_values(self, values: NDArrayFloat) -> None:
        """Set the observed values."""
        self._values = values

    @property
    def mean_type(self) -> MeanType:
        """Return the mean type used to interpolate values over several grid cells."""
        return self._mean_type

    @mean_type.setter
    def mean_type(self, value: Optional[MeanType]) -> None:
        """Set the mean type used to interpolate values over several grid cells."""
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

    def set_perturbations(self, pvals: NDArrayFloat) -> None:
        """Set the values perturbations (see ensemble smoothers)."""
        if self.values.size != pvals.size:
            raise ValueError(
                "perturbations size should match observation values size !"
            )
        self.perturbations = pvals


# new type
Observables = Union[Observable, Sequence[Observable]]


def update_perturbation_values(observables: Observables, pvals: NDArrayFloat) -> None:
    """Update the perturbation values for the given observables instances."""
    first_index = 0
    for obs in object_or_object_sequence_to_list(observables):
        last_index = first_index + obs.values.size
        # Update perturbations
        obs.perturbations = pvals[first_index:last_index]


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
    model: ForwardModel, state_variable: StateVariable, sp: Optional[int] = None
) -> NDArrayFloat:
    if state_variable == StateVariable.CONCENTRATION:
        if sp is None:
            raise ValueError("sp cannot be None for concentrations")
        return model.tr_model.mob[sp]
    if state_variable == StateVariable.HEAD:
        return model.fl_model.head
    if state_variable == StateVariable.PRESSURE:
        return model.fl_model.pressure
    if state_variable == StateVariable.GRADE:
        if sp is None:
            raise ValueError("sp cannot be None for grades")
        return model.tr_model.immob[sp]
    if state_variable == StateVariable.PERMEABILITY:
        return model.fl_model.permeability
    if state_variable == StateVariable.POROSITY:
        return model.tr_model.porosity
    if state_variable == StateVariable.DIFFUSION:
        return model.tr_model.diffusion
    if state_variable == StateVariable.DENSITY:
        return model.tr_model.density
    if state_variable == StateVariable.STORAGE_COEFFICIENT:
        return model.fl_model.storage_coefficient
    else:
        raise ValueError(
            f'"{state_variable}" is not a valid state variable or parameter type!'
        )


def get_observables_values_as_1d_vector(
    observables: Observables,
    max_obs_time: Optional[float] = None,
) -> NDArrayFloat:
    """
    Return the values of all given observables as a 1D vector.

    Note
    ----
    The observation values are sorted first by Observable instance (observation
    location) and by ascending time at a second level.

    Parameters
    ----------
    observables
        Sequence of observable instances.
    max_obs_time : Optional[float], optional
        Maximum time for which to consider an observation value, by default None.
    """
    return np.hstack(
        [
            get_sorted_observable_values(obs, max_obs_time)
            for obs in object_or_object_sequence_to_list(observables)
        ]
    ).ravel()


def get_observables_uncertainties_as_1d_vector(
    observables: Observables,
    max_obs_time: Optional[float] = None,
) -> NDArrayFloat:
    """Return the uncertainties of all observables as a 1D vector.

    Note
    ----
    The observation values are sorted first by Observable instance (observation
    location) and by ascending time at a second level.

    Parameters
    ----------
    observables
        Sequence of observable instances.
    max_obs_time : Optional[float], optional
        Maximum time for which to consider an observation value, by default None.

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
    Get the calculated times before and after the observation times.

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
    _den = den.copy()
    _den[den == 0] = 1.0

    weights_before = 1 - (obs_times - simu_times[before_idx]) / _den
    weights_before[before_idx < 0] = 0.0  # Handle obs at time zero.
    weights_after = 1 - (simu_times[after_idx] - obs_times) / _den

    weights_before[den == 0] = 1.0
    weights_after[den == 0] = 0.0

    return weights_before, weights_after


def get_values_matching_node_indices(
    node_indices: NDArrayInt, input_values: NDArrayFloat
) -> NDArrayFloat:
    r"""
    Return the values for the given node_indices with shape.

    The output shape is (:math:`\lvert \mathrm{node_indices} \rvert|, [nt]),
    where :math:`\lvert . \rvert` means cardinality.

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
    Get an array of interpolated values that match the observation times.

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
        Maximum time for which to consider an observation value, by default None.

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

    field = get_array_from_state_variable(model, obs.state_variable, obs.sp)
    obs_times = get_sorted_observable_times(obs, max_obs_time)

    if len(field.shape) == 2:
        _field = field.reshape((*field.shape, 1))
    else:
        _field = field
    _simu_values = get_mean_values_for_last_axis(
        get_values_matching_node_indices(obs.node_indices, _field),
        mean_type=obs.mean_type,
        weights=None,
    )
    # control parameter -> not varying in time
    if len(field.shape) == 2:
        # The interpolated value is the same for all time
        return np.repeat(_simu_values, obs_times.size)
    # state variable varying in time
    return get_interp_simu_values_matching_obs_times(
        obs_times, simu_times, _simu_values
    )


def get_adjoint_sources_for_obs(
    model: ForwardModel,
    obs: Observable,
    n_obs: int,
    max_obs_time: Optional[float] = None,
) -> NDArrayFloat:
    r"""
    Get the adjoint sources for a given observable instance.

    The objective function with respect to the vector of observed
    state variables and control parameters :math:`(\mathbf{d})`
    is defined as:

    .. math::
        \mathcal{J}(\mathbf{d}_{\mathrm{calc}}) = \dfrac{1}{2N} \sum_{n=0}^{N}
        \left(\dfrac{d_{\mathrm{obs}}^{t}
        - d_{\mathrm{calc}}^{t}}{\sigma_{\mathrm{obs}}^{t}} \right)^{2}
        = \dfrac{1}{2N} \sum_{n=0}^{N}
        \left(\dfrac{d_{\mathrm{obs}}^{t}
        - \left(\omega_{n}av(d_{\mathrm{calc}}^{n})
        + \omega_{n+1}av(d_{\mathrm{calc}}^{n+1})\right)}{
            \sigma_{\mathrm{obs}}^{t}} \right)^{2}

    with :math:`d_{\mathrm{obs}}^{t}` an observation at a time :math:`t`
    comprised between simulation iterations :math:`n` and :math:`n+1`,
    :math:`N = \lvert \mathbf{d}_{\mathrm{obs}} \rvert`
    the number of observation points, and :math:`\omega` the weights for the linear
    interpolation of the calculated value which read:

    - :math:`\omega_{n} = 1 - \dfrac{t - t(n)}{t(n+1) - t(n)}`
    - :math:`\omega_{n+1} =  1 - \dfrac{(t(n+1) - t)}{t(n+1) - t(n)}`

    And :math:`av` a spatial averaging operator required if the observation is done on
    several grid cells. Consequently, the objective function depends both on
    :math:`d_{\mathrm{calc}}^{n}` and :math:`d_{\mathrm{calc}}^{n+1}`,
    and the derivatives read:

    .. math::

        \begin{eqnarray}
        \dfrac{\partial\mathcal{J}}{\partial d_{\mathrm{calc}}^{n}} & = &
        \dfrac{\partial}{\partial d_{\mathrm{calc}}^{n}} \left(\dfrac{1}{2 \lvert
        \mathbf{d}_{\mathrm{obs}} \rvert } \sum_{n=0}^{N}
        \left(\dfrac{d_{\mathrm{obs}}^{t} - \left(\omega_{n}av(d_{\mathrm{calc}}^{n})
        + \omega_{n+1}av(d_{\mathrm{calc}}^{n+1})\right)}{
            \sigma_{\mathrm{obs}}^{t}} \right)^{2}\right) \\
        & = &  - \dfrac{\omega_{n}}{N }
        \dfrac{\partial av(d_{\mathrm{calc}}^{n})}{\partial d_{\mathrm{calc}}^{n}}
        \dfrac{d_{\mathrm{obs}}^{t} - \left(\omega_{n}av(d_{\mathrm{calc}}^{n})
        + \omega_{n+1}av(d_{\mathrm{calc}}^{n+1})\right)}{
            \left(\sigma_{\mathrm{obs}}^{t}\right)^{2}}
        \end{eqnarray}

    And

    .. math::
        \dfrac{\partial\mathcal{J}}{\partial d_{\mathrm{calc}}^{n+1}} =
        - \dfrac{\omega_{n+1}}{N}
        \dfrac{\partial av(d_{\mathrm{calc}}^{n+1})}{\partial d_{\mathrm{calc}}^{n}}
        \dfrac{d_{\mathrm{obs}}^{t} - \left(\omega_{n}av(d_{\mathrm{calc}}^{n})
        + \omega_{n+1}av(d_{\mathrm{calc}}^{n+1})\right)}{
            \left(\sigma_{\mathrm{obs}}^{t}\right)^{2}}

    Parameters
    ----------
    model : ForwardModel
        _description_
    obs : Observable
        _description_
    max_obs_time : Optional[float], optional
        Maximum time for which to consider an observation value, by default None.
    n_obs: float
        The number of observation point to consider.

    Returns
    -------
    NDArrayFloat
        The adjoint sources for the given Observable instance.
    """

    simu_times = np.cumsum([0] + model.time_params.ldt)
    if max_obs_time is not None:
        max_obs_time = min(np.max(simu_times), max_obs_time)
    else:
        max_obs_time = np.max(simu_times)

    obs_times = get_sorted_observable_times(obs, max_obs_time)
    obs_values = get_sorted_observable_values(obs, max_obs_time)
    obs_std = get_sorted_observable_uncertainties(obs, max_obs_time)
    simu_values = get_simulated_values_matching_obs(model, obs, max_obs_time)

    # 1) Taking into account the derivative linked with the values averaging over the
    # the grid (observations defined on more than one mesh)
    field = get_array_from_state_variable(model, obs.state_variable, obs.sp)
    _averaging_derivative = get_mean_values_gradient_for_last_axis(
        get_values_matching_node_indices(obs.node_indices, field),
        mean_type=obs.mean_type,
        weights=None,
    )
    adj_src = np.zeros(field.shape)

    # Location in the grid
    X, Y, _ = node_number_to_indices(
        obs.node_indices, nx=model.geometry.nx, ny=model.geometry.ny
    )

    # 2) Taking into account the derivative linked with the values time interpolation
    # (observations defined between two times of the simulation)

    # no time dimension, so normally, the adj_src are for the parameter at t=0
    # (ex: permeability)
    if len(adj_src.shape) == 2:
        for n in range(len(obs_times)):
            adj_src[X, Y] += (
                _averaging_derivative  # in this case
                * (simu_values - obs_values)[n]
                / (obs_std[n] ** 2)
            )
    else:
        # Otherwise, we derive the weights of the linear interpolation
        idx_before, idx_after = get_times_idx_before_after_obs(obs_times, simu_times)
        weights_before, weights_after = get_weights(
            obs_times, simu_times, idx_before, idx_after
        )
        for n in range(len(obs_times)):
            # Note: ici, on a simu_values = w1 * av(c(n+1)) + w2 * av(c(n))
            adj_src[X, Y, idx_before[n]] += (
                _averaging_derivative[:, idx_before[n]].ravel("F")
                * weights_before[n]
                * (simu_values - obs_values)[n]
                / (obs_std[n] ** 2)
            )
            adj_src[X, Y, idx_after[n]] += (
                _averaging_derivative[:, idx_after[n]].ravel("F")
                * weights_after[n]
                * (simu_values - obs_values)[n]
                / (obs_std[n] ** 2)
            )

    # Return the adjoint sources
    return adj_src


def get_predictions_matching_observations(
    model: ForwardModel,
    observables: Observables,
    max_obs_time: Optional[float] = None,
) -> NDArrayFloat:
    """
    Return the 1D vector of predictions matching the observations.

    Parameters
    ----------
    model : ForwardModel
        _description_
    observables : Observables
        _description_
    max_obs_time : Optional[float], optional
        Maximum time for which to consider an observation value, by default None.

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
