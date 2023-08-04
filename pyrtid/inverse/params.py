"""
Class to define parameters that will be estimated using hytec-python-auto-hm.

@author: acollet
"""
from __future__ import annotations

import json
import sys
import warnings
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numdifftools as nd
import numpy as np

from pyrtid.forward import ForwardModel
from pyrtid.inverse.regularization import Regularizator
from pyrtid.utils.enum import StrEnum
from pyrtid.utils.finite_differences import is_all_close
from pyrtid.utils.spatial_filters import Filter
from pyrtid.utils.types import (
    NDArrayBool,
    NDArrayFloat,
    NDArrayInt,
    object_or_object_sequence_to_list,
)


def identify_function(x: NDArrayFloat) -> NDArrayFloat:
    """Return x untransformed (f(x) = x)."""
    return x


def one(x: NDArrayFloat) -> NDArrayFloat:
    """Return 1.0, whatever the input."""
    return np.ones(x.shape)


class ParameterName(StrEnum):
    """Name of supported parameters."""

    DIFFUSION = "diffusion"
    POROSITY = "porosity"
    PERMEABILITY = "permeability"
    INITIAL_CONCENTRATION = "initial_concentration"
    INITIAL_MINERAL_GRADE = "initial_mineral_grade"


@dataclass
class AdjustableParameter:
    """
    Represent an adjustable parameter for hytec-python-auto-hm.

    It might represent a unique numerical value or an entire field.

    Attributes
    ----------
    name : ParameterType
        The parameter name, such as 'Porosity'.
    values: NDArrayFloat, optional
        The values of the parameter field. The default is an empty array.
        Note: value is a 2D ndarray, which possesses all values, including not
        adjusted ones.
    lbound : Optional[float]
        The lower bound for the parameter.The default is -1e20.
    ubound : Optional[float]
        The upper bound for the parameter. The default is 1e20.
    regularizator: List[Regularizator]
        List of regularization to apply to the parameter.
    preconditioner: Callable, optional
        Parameter pre-transformation (variable change for the solver). The default
        is the identity function: f(x) = x.
    preconditioner_1st_derivative: Callable, optional
        Parameter pre-transformation first order derivative.
        The default is 1.0 (the first derivative of the identity function).
    backconditioner: Callable, optional
        Parameter back-transformation (variable change for the solver). The default
        is the identity function: f(x) = x.
    span : Optional[Union[slice, NDArrayBool]
        The span over which the parameter applies.
    archived_adjoint_gradients: List[NDArrayFloat]
        List of successive adjoint gradients computed while optimizing.
    archived_fd_gradients: List[NDArrayFloat]
        List of successive finite difference gradients computed while optimizing.
    """

    __slots__ = [
        "name",
        "values",
        "_lbound",
        "_ubound",
        "regularizators",
        "preconditioner",
        "preconditioner_1st_derivative",
        "backconditioner",
        "span",
        "filters",
        "archived_values",
        "archived_adjoint_gradients",
        "archived_fd_gradients",
        "reg_weight",
    ]

    def __init__(
        self,
        name: ParameterName,
        values: NDArrayFloat = np.array([]),
        lbound: float = -1e20,
        ubound: float = 1e20,
        regularizators: Optional[List[Regularizator]] = None,
        preconditioner: Callable = identify_function,
        preconditioner_1st_derivative: Callable = one,
        backconditioner: Callable = identify_function,
        span: Union[NDArrayInt, Tuple[slice, slice], NDArrayBool] = (
            slice(None),
            slice(None),
        ),
        filters: Optional[List[Filter]] = None,
    ) -> None:
        self.name = name
        self.values = values
        self.lbound = lbound
        self.ubound = ubound
        self.regularizators = regularizators if regularizators is not None else []
        self.preconditioner = preconditioner
        self.preconditioner_1st_derivative = preconditioner_1st_derivative
        self.backconditioner = backconditioner
        self.span = span
        self.filters = filters if filters is not None else []

        self._test_preconditioner()
        self._test_bounds_consistency()
        self.archived_values: List[NDArrayFloat] = []
        self.archived_adjoint_gradients: List[NDArrayFloat] = []
        self.archived_fd_gradients: List[NDArrayFloat] = []
        self.reg_weight: float = 0

        for regularizator in self.regularizators:
            if not isinstance(regularizator, Regularizator):
                raise ValueError("Expect a regulariator instance !")

    @property
    def lbound(self) -> float:
        """Return the lower bound value."""
        return self._lbound

    @lbound.setter
    def lbound(self, _value: Union[int, float, NDArrayInt, NDArrayFloat]) -> None:
        """Set the lower bound value."""
        self._lbound = float(_value)

    @property
    def ubound(self) -> float:
        """Return the upper bound value."""
        return self._ubound

    @ubound.setter
    def ubound(self, _value: Union[int, float, NDArrayInt, NDArrayFloat]) -> None:
        """Set the upper bound value."""
        self._ubound = float(_value)

    @property
    def size(self) -> int:
        """Return the size (number of values) of the parameter."""
        return self.values.size

    @property
    def size_adjusted_values(self) -> int:
        """Return the number of adjusted values of the parameter."""
        if self.size == 0:
            return 0
        return self.values[self.span].size

    def _test_preconditioner(self) -> None:
        """Test if the backconditioner is the inverse of the preconditioner."""
        test_data: NDArrayFloat = np.linspace(self.ubound, self.lbound, num=50)

        # 1) check if the back and pre-conditioner match
        if not np.allclose(
            test_data, self.backconditioner(self.preconditioner(test_data))
        ):
            raise ValueError(
                "The given backconditioner does not match the preconditioner! or"
                " does not match the given bounds: "
                f"lbound = {self.lbound}, ubound = {self.ubound}"
            )
        # 2) check by finite difference if the pre-conditioner 1st derivative is correct
        valid: bool = is_all_close(
            self.preconditioner_1st_derivative(test_data),
            # Finite difference differentiation
            nd.Derivative(self.preconditioner, n=1, step=sys.float_info.epsilon * 1e8)(
                test_data
            ),  # type: ignore
        )
        if not valid:
            warnings.warn(
                UserWarning(
                    "The given preconditioner 1st derivative check by finite "
                    "difference failed ! The given function might no match the "
                    "preconditioner or is not defined on the given bounds: "
                    f"lbound = {self.lbound}, ubound = {self.ubound}."
                    " Note that it can also append if bounds are very huge number, "
                    "e.g., 1e20."
                )
            )

    @property
    def max_value(self) -> float:
        """Return the max of the values."""
        try:
            return float(np.max(self.values))
        except ValueError:
            return np.nan

    @property
    def min_value(self) -> float:
        """Return the min of the values."""
        try:
            return float(np.min(self.values))
        except ValueError:
            return np.nan

    def _test_bounds_consistency(self) -> None:
        """Test that ubound > lbound."""
        if self.lbound >= self.ubound:
            raise ValueError("lbound should be strictly inferior to ubound.")

    def __str__(self) -> str:
        """Represent the class object as a string."""
        return json.dumps(
            {
                "name": self.name,
                "size": self.size,
                "size_adjsuted_vaues": self.size_adjusted_values,
                "min_value": self.min_value,
                "max_value": self.max_value,
                "lbound": self.lbound,
                "ubound": self.ubound,
                "preconditioner": self.preconditioner,
                "preconditioner_1st_derivative": self.preconditioner_1st_derivative,
                "backconditioner": self.backconditioner,
                "span": self.span,
            },
            indent=4,
            sort_keys=False,
            default=str,
        ).replace("null", "None")

    def __eq__(self, other) -> bool:
        """
        Define equivalence between two AdjustableParameter instances.

        Two instances are considered equivalent if their name and source
        files are identical.
        """
        if isinstance(other, self.__class__):
            return other.name == self.name
        return False

    def __ne__(self, other) -> bool:
        """Define non equivalence between two AdjustableParameter instances."""
        return not self == other

    def update(self, other: AdjustableParameter) -> None:
        """Update the attributes with other's."""
        for attr in other.__slots__:
            val = other.__getattribute__(attr)
            if val is not None:
                setattr(self, attr, val)

    def get_sliced_field(
        self, input_field: NDArrayFloat, is_preconditioned: bool = False
    ) -> NDArrayFloat:
        """Return a slice of the input data field."""
        if is_preconditioned:
            return self.preconditioner(input_field[self.span])
        return input_field[self.span]

    def get_preconditioned_1st_derivative_sliced_field(
        self, input_field: NDArrayFloat
    ) -> NDArrayFloat:
        """Return a preconditioned slice of the input data field."""
        return self.preconditioner_1st_derivative(input_field[self.span])

    def get_values_from_model_field(self, input_field: NDArrayFloat) -> None:
        """Update the parameter values from the given input field."""
        # self.values = input_field[self.span]
        self.values = input_field

    def update_values_with_vector(
        self, update_values: NDArrayFloat, is_preconditioned: bool = False
    ) -> None:
        """Update the values attribute from the given vector.

        Parameters
        ----------
        update_values : NDArrayFloat
            The values used for update (preconditioned).
        """
        if is_preconditioned:
            func: Callable = self.backconditioner
        else:
            func = identify_function
        tmp = func(np.reshape(update_values.copy(), self.values[self.span].shape))
        tmp = np.where(tmp < self.lbound, self.lbound, tmp)
        self.values[self.span] = np.where(tmp > self.ubound, self.ubound, tmp)

    def update_field_with_param_values(self, field_to_update: NDArrayFloat) -> None:
        """Update the input field with the Adjustable parameter current values."""
        # sub_field = field_to_update[self.span]
        # values = np.reshape(self.values, sub_field.shape)
        field_to_update[self.span] = self.values[self.span]

    def get_bounds(self, is_preconditioned: bool = False) -> NDArrayFloat:
        """
        Return a 2*n bounds matrix.

        Parameters
        ----------
        is_preconditioned: bool, optional
            Whether to return preconditioned bounds or not. The default is False.

        Note
        ----
        It takes into account the preconditioning step and the fact that the
        preconditioner might no be strictly increasing function.
        """
        bounds = np.array([self.lbound, self.ubound])
        if is_preconditioned:
            bounds = self.preconditioner(bounds)
        return np.concatenate(
            [
                np.full((1, self.size_adjusted_values), min(bounds)),
                np.full((1, self.size_adjusted_values), max(bounds)),
            ]
        ).T

    def get_regularization_loss_function(self) -> float:
        """
        Return the regularization objective function for the parameter.

        Note
        ----
        A ratio is applied.
        """
        _sum: float = 0.0
        for reg in self.regularizators:
            values: NDArrayFloat = self.values.copy()
            if reg.is_preconditioned:
                values: NDArrayFloat = self.preconditioner(values)
            _sum += reg.loss_function(values)
        return _sum

    def get_regularization_loss_function_gradient(
        self,
    ) -> Union[NDArrayFloat, float]:
        """Return the regularization objective function gradient."""
        grad: NDArrayFloat = np.zeros(self.values.shape)
        for reg in self.regularizators:
            values: NDArrayFloat = self.values.copy()
            if reg.is_preconditioned:
                values: NDArrayFloat = self.preconditioner(values)
            grad += reg.loss_function_gradient(values)
        return grad


def get_parameter_values_from_model(
    model: ForwardModel, param: AdjustableParameter
) -> NDArrayFloat:
    """
    Get the adjusted parameter values in the model.

    Parameters
    ----------
    param : AdjustableParameter
        The adjusted parameter.

    Raises
    ------
    ValueError
        If the given parameter is not supported.

    Returns
    -------
    NDArrayFloat
        The required parameter values form the model.

    """
    if param.name not in ParameterName.to_list():
        raise ValueError(
            f"{param.name} is not an adjustable parameter !\n"
            f"Supported parameters are {ParameterName.to_list()}"
        )
    if param.name == ParameterName.DIFFUSION:
        return model.tr_model.diffusion
    if param.name == ParameterName.POROSITY:
        return model.tr_model.porosity
    if param.name == ParameterName.PERMEABILITY:
        return model.fl_model.permeability
    if param.name == ParameterName.INITIAL_CONCENTRATION:
        return model.tr_model.conc[:, :, 0]
    if param.name == ParameterName.INITIAL_MINERAL_GRADE:
        return model.tr_model.grade[:, :, 0]
    raise (NotImplementedError("Please contact the developer to handle this issue."))


def update_parameters_from_model(
    model: ForwardModel,
    parameters_to_adjust: Union[AdjustableParameter, Sequence[AdjustableParameter]],
) -> None:
    """Update adjusted parameters from model."""
    for param in object_or_object_sequence_to_list(parameters_to_adjust):
        # check if the values are empty
        if param.values.size == 0:
            param.values = get_parameter_values_from_model(model, param)


def get_parameters_values_from_model(
    model: ForwardModel,
    params: Union[AdjustableParameter, Sequence[AdjustableParameter]],
    is_preconditioned: bool = False,
) -> NDArrayFloat:
    """
    Return a 1D vector of stacked values of parameters from the model.

    Parameters
    ----------
    is_preconditioned: bool, optional
        Whether to return preconditioned bounds or not. The default is False.

    Note
    ----
    The order of values in the returned vector is the same as the given parameters.
    """
    data = []
    for param in object_or_object_sequence_to_list(params):
        data.append(
            param.get_sliced_field(
                get_parameter_values_from_model(model, param), is_preconditioned
            )
        )
    # Concatenate the arrays and make it 1D
    return np.vstack(data).ravel()


def get_1st_derivative_preconditoned_parameters_values_from_model(
    model: ForwardModel,
    params: Union[AdjustableParameter, Sequence[AdjustableParameter]],
) -> NDArrayFloat:
    """
    Return a 1D vector of preconditioned inverted model parameters derivative.

    Note
    ----
    The preconditioning depends on whether the user wanted to use one or not.
    """
    data = []
    for param in object_or_object_sequence_to_list(params):
        data.append(
            param.get_preconditioned_1st_derivative_sliced_field(
                get_parameter_values_from_model(model, param)
            )
        )
    # Concatenate the arrays and make it 1D
    return np.vstack(data).ravel()


def update_model_with_parameters_values(
    model: ForwardModel,
    parameters_values: NDArrayFloat,
    params: Union[AdjustableParameter, Sequence[AdjustableParameter]],
    is_preconditioned: bool = False,
    is_to_save: bool = False,
) -> None:
    """
    Update the params and the model with the given preconditoned values `x`.

    Note
    ----
    x is a preconditoned vector.
    """
    # The parameters should be read in the same order as in the method
    # get_adjusted_params_vector
    first_index = 0
    for param in object_or_object_sequence_to_list(params):
        last_index = first_index + param.size
        # Update values in param
        param.update_values_with_vector(
            parameters_values[first_index:last_index], is_preconditioned
        )
        # Update the model from param
        param.update_field_with_param_values(
            get_parameter_values_from_model(model, param)
        )
        first_index = last_index
        # Store the values
        if is_to_save:
            param.archived_values.append(param.values.copy())


def get_parameters_bounds(
    params: Union[AdjustableParameter, Sequence[AdjustableParameter]],
    is_preconditioned: bool = False,
) -> np.ndarray:
    """Return a 2xn bounds matrix, n being the number of parameters values inverted.

    Parameters
    ----------
    is_preconditioned: bool, optional
        Whether to return preconditioned bounds or not. The default is False.

    """
    return np.concatenate(
        [
            param.get_bounds(is_preconditioned)
            for param in object_or_object_sequence_to_list(params)
        ]
    )


def get_backconditioned_adj_gradient(
    param: AdjustableParameter, index: int
) -> NDArrayFloat:
    return param.archived_adjoint_gradients[
        index
    ] / param.preconditioner_1st_derivative(param.values)


def get_backconditioned_fd_gradient(
    param: AdjustableParameter, index: int
) -> NDArrayFloat:
    return param.archived_fd_gradients[index] / param.preconditioner_1st_derivative(
        param.values
    )


def get_gridded_archived_gradients(
    param, is_adjoint: bool, is_use_span: bool = False
) -> NDArrayFloat:
    """Return an array (nx, ny, nt) of gridded archived gradients.

    Note
    ----
    The non adjusted grid cells are given as NaNs.

    Parameters
    ----------
    is_adjoint : bool
        Whether to use adjoint or finite differences gradients.
    is_use_span: bool, optional
        Whether to apply the parameter span selection or not. The default is False.

    Returns
    -------
    NDArrayFloat
        Array (nx, ny, nt) of gridded archived gradients.
    """
    if is_adjoint:
        gradients: List[NDArrayFloat] = param.archived_adjoint_gradients
    else:
        gradients: List[NDArrayFloat] = param.archived_fd_gradients
    out: NDArrayFloat = np.empty((*param.values.shape, len(gradients)))
    out[:] = np.nan
    for i, vals in enumerate(gradients):
        if is_use_span:
            out[:, :, i][param.span] = vals.reshape(*param.values.shape)[param.span]
        else:
            out[:, :, i] = vals.reshape(*param.values.shape)
    return out


def get_param_values(
    param: AdjustableParameter,
    is_use_span: bool = True,
    is_preconditioned: bool = False,
) -> NDArrayFloat:
    """Return a slice of the input data field."""
    _values = param.values
    if is_use_span:
        _values = _values[param.span]
    if is_preconditioned:
        return param.preconditioner(_values)
    return _values
