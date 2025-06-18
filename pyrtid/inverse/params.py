"""
Class to define parameters that will be estimated using hytec-python-auto-hm.

@author: acollet
"""

from __future__ import annotations

import copy
import json
import logging
from typing import Callable, List, Optional, Sequence, Union

import numpy as np
import scipy as sp

from pyrtid.forward import ForwardModel
from pyrtid.inverse.obs import StateVariable, get_array_from_state_variable
from pyrtid.inverse.regularization import (
    ConstantRegWeight,
    Regularizator,
    RegWeightUpdateStrategy,
)
from pyrtid.utils import (
    NDArrayFloat,
    NDArrayInt,
    StrEnum,
    object_or_object_sequence_to_list,
)
from pyrtid.utils.preconditioner import (
    GradientScalerConfig,
    NoTransform,
    Preconditioner,
)
from pyrtid.utils.spatial_filters import Filter


def identify_function(x: NDArrayFloat) -> NDArrayFloat:
    """Return x untransformed (f(x) = x)."""
    return x


class ParameterName(StrEnum):
    """Type of parameter invertible."""

    INITIAL_CONCENTRATION = "concentration"
    DIFFUSION = "diffusion"
    INITIAL_HEAD = "head"
    INITIAL_GRADE = "grade"
    PERMEABILITY = "permeability"
    POROSITY = "porosity"
    INITIAL_PRESSURE = "pressure"
    STORAGE_COEFFICIENT = "storage_coefficient"
    DISPERSIVITY = "dispersivity"


PARAM_TO_STATE_VAR = {
    ParameterName.INITIAL_CONCENTRATION: StateVariable.CONCENTRATION,
    ParameterName.DIFFUSION: StateVariable.DIFFUSION,
    ParameterName.INITIAL_HEAD: StateVariable.HEAD,
    ParameterName.INITIAL_GRADE: StateVariable.GRADE,
    ParameterName.PERMEABILITY: StateVariable.PERMEABILITY,
    ParameterName.POROSITY: StateVariable.POROSITY,
    ParameterName.INITIAL_PRESSURE: StateVariable.PRESSURE,
    ParameterName.STORAGE_COEFFICIENT: StateVariable.STORAGE_COEFFICIENT,
    ParameterName.DISPERSIVITY: StateVariable.DISPERSIVITY,
}


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
    lbounds : Union[float, NDArrayFloat]
        The lower bounds for the parameter.
    ubounds : Union[float, NDArrayFloat]
        The upper bounds for the parameter.
    regularizator: List[Regularizator]
        List of regularization to apply to the parameter.
    preconditioner: Preconditioner
        Parameter pre-transformation operator (variable change for the solver).
        The default is the identity function: f(x) = x, which means no change
        is made.
    filters : List[Filter]
        List of filters to apply to the gradient.
    grad_adj_history: List[NDArrayFloat]
        List of successive adjoint gradients (w.r.t preconditioned parameter values)
        computed while optimizing.
    grad_adj_raw_history: List[NDArrayFloat]
        List of successive adjoint gradients (w.r.t non-preconditioned (raw) parameter
        values) computed while optimizing.
    grad_fd_history: List[NDArrayFloat]
        List of successive finite difference gradients computed while optimizing.
    jacvec_fsm_history: List[NDArrayFloat]
        List of successive jacobian dot vectors obtained with the forward sensitivity
        method (w.r.t preconditioned parameter values) computed while optimizing.
    jacvec_fsm_raw_history: List[NDArrayFloat]
        List of successive jacobian dot vectors obtained with the forward sensitivity
        method (w.r.t non-preconditioned (raw) parameter values) computed while
        optimizing.
    jacvec_fd_history: List[NDArrayFloat]
        List of successive finite difference jacobian dot vectors computed
        while optimizing.
    sp: Optional[int]
        Index of the species.
    reg_weight: float
        Current regularization weight (parameter).
    reg_weight_update_strategy: RegWeightUpdateStrategy
        Strategy for the regularization parameter (weight) update.
        This has no effect if no regularizator is defined.
    reg_weight_history: List[float]
        Successive regularization parameter used in the optimization process.
    loss_reg_history: List[float]
        Successive regularization loss obtained in the optimization process.
    """

    __slots__ = [
        "name",
        "values",
        "_lbounds",
        "_ubounds",
        "regularizators",
        "preconditioner",
        "filters",
        "archived_values",
        "grad_adj_history",
        "grad_adj_raw_history",
        "grad_fd_history",
        "jacvec_fsm_history",
        "jacvec_fsm_raw_history",
        "jacvec_fd_history",
        "reg_weight_update_strategy",
        "reg_weight_history",
        "loss_reg_history",
        "sp",
        "gradient_scaler_config",
    ]

    def __init__(
        self,
        name: ParameterName,
        values: Optional[NDArrayFloat] = None,
        lbounds: Union[float, NDArrayFloat] = -np.inf,
        ubounds: Union[float, NDArrayFloat] = np.inf,
        regularizators: Optional[Union[Regularizator, List[Regularizator]]] = None,
        preconditioner: Preconditioner = NoTransform(),
        filters: Optional[List[Filter]] = None,
        sp: Optional[int] = None,
        reg_weight_update_strategy: RegWeightUpdateStrategy = (ConstantRegWeight(1.0)),
        gradient_scaler_config: Optional[GradientScalerConfig] = None,
    ) -> None:
        """
        Initialize the instance.

        Parameters
        ----------
        name : ParameterType
            The parameter name, such as 'Porosity'.
        values: NDArrayFloat, optional
            The values of the parameter field. The default is an empty array.
            Note: value is a 2D ndarray, which possesses all values, including not
            adjusted ones.
        lbounds : Union[float, NDArrayFloat]
            The lower bound for the parameter.The default is -np.inf.
        ubounds : Union[float, NDArrayFloat]
            The upper bound for the parameter. The default is np.inf.
        regularizator: List[Regularizator]
            List of regularization to apply to the parameter.
        preconditioner: Preconditioner.
            Parameter pre-transformation instance (variable change for the solver).
            The default is the identity function: f(x) = x.
        filters : Optional[List[Filter]], optional
            List of filters to apply to the gradient. by default None
        sp: Optional[int]
            Index of the species to optimize. Mandtory if a concentration or a grade is
            being inverted. The default is None.
        reg_weight_update_strategy: RegWeightUpdateStrategy
            Strategy for the regularization parameter (weight) update.
            This has no effect if no regularizator is defined. The default is no_update.
        gradient_scaler_config: Optional[GradientScalerConfig]
            Condifguration for the gradient scaling approach with L-BFGS-B.
            This configuration is used only if the L-BFGS-B optimizer is used for
            inversion. The default is None.

        Raises
        ------
        ValueError
            In case of issue with the regularizators or the preconditoning.
        """
        self.name = name
        self.values = values if values is not None else np.array([])
        self.lbounds = lbounds
        self.ubounds = ubounds
        self.regularizators = (
            object_or_object_sequence_to_list(regularizators)
            if regularizators is not None
            else []
        )
        self.preconditioner: Preconditioner = preconditioner
        self.filters = filters if filters is not None else []

        self._test_bounds_consistency()

        for regularizator in self.regularizators:
            if not isinstance(regularizator, Regularizator):
                raise ValueError("Expect a regularizator instance !")

        if (
            self.name
            in [ParameterName.INITIAL_CONCENTRATION, ParameterName.INITIAL_GRADE]
            and sp is None
        ):
            raise ValueError(
                "sp must be provided when optimizing an initial "
                "grade or an initial concentration!"
            )

        if sp is None:
            self.sp: int = 0
        else:
            self.sp = sp

        self.reg_weight_update_strategy = reg_weight_update_strategy
        # initialize internal lists
        self.init_state()

        self.gradient_scaler_config: Optional[GradientScalerConfig] = (
            gradient_scaler_config
        )

    def init_state(self) -> None:
        """Initialize the internal state (lists, comptors, etc.)"""
        self.archived_values: List[NDArrayFloat] = []
        self.grad_adj_history: List[NDArrayFloat] = []  # preconditioned
        self.grad_adj_raw_history: List[NDArrayFloat] = []  # non-preconditioned
        self.grad_fd_history: List[NDArrayFloat] = []
        self.jacvec_fsm_history: List[NDArrayFloat] = []  # preconditioned
        self.jacvec_fsm_raw_history: List[NDArrayFloat] = []  # non-preconditioned
        self.jacvec_fd_history: List[NDArrayFloat] = []
        self.reg_weight_history: List[float] = []
        self.loss_reg_history: List[float] = []

    @property
    def lbounds(self) -> NDArrayFloat:
        """Return the lower bound values."""
        return self._lbounds

    @lbounds.setter
    def lbounds(self, _values: Union[int, float, NDArrayInt, NDArrayFloat]) -> None:
        """Set the lower bound values."""
        self._lbounds = np.array(_values, dtype=np.float64)

    @property
    def ubounds(self) -> NDArrayFloat:
        """Return the upper bound values."""
        return self._ubounds

    @ubounds.setter
    def ubounds(self, _values: Union[int, float, NDArrayInt, NDArrayFloat]) -> None:
        """Set the upper bound value."""
        self._ubounds = np.array(_values, dtype=np.float64)

    @property
    def size(self) -> int:
        """Return the size (number of values) of the parameter."""
        return self.values.size

    @property
    def size_preconditioned_values(self) -> int:
        """Return the number of values of the preconditioned parameter."""
        if self.size == 0:
            return 0
        return self.preconditioner(self.values.ravel("F")).size

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

    @property
    def reg_weight(self) -> float:
        """Regularization weight."""
        return self.reg_weight_update_strategy.reg_weight

    @property
    def is_scale_logarithmically(self) -> bool:
        """Return whether the parameter scales logarithmically."""
        if self.name in [
            ParameterName.DIFFUSION,
            ParameterName.STORAGE_COEFFICIENT,
            ParameterName.PERMEABILITY,
        ]:
            return True
        return False

    def _test_bounds_consistency(self) -> None:
        """Test that ubounds > lbounds."""
        # test that size is equal + that lbounds <= ubounds
        try:
            # use negative sign to get the <=
            np.testing.assert_array_less(-self.ubounds, -self.lbounds)
        except AssertionError as e:
            raise ValueError(
                "lbounds should be strictly inferior to "
                "ubounds and have the same shape!"
            ) from e
        if (
            self.lbounds < self.preconditioner.LBOUND_RAW
            or self.ubounds > self.preconditioner.UBOUND_RAW
        ):
            raise ValueError(
                "The provided parameter bounds do not match with the "
                "range of values supported by the preconditioner: "
                f"[{self.preconditioner.LBOUND_RAW}, {self.preconditioner.UBOUND_RAW}]"
            )

    def __str__(self) -> str:
        """Return a string representation of the instance."""
        return json.dumps(
            {
                "name": self.name,
                "size": self.size,
                "size_adjsuted_vaues": self.size_preconditioned_values,
                "min_value": self.min_value,
                "max_value": self.max_value,
                "min(lbounds)": np.min(self.lbounds),
                "max(ubounds)": np.max(self.ubounds),
                "preconditioner": self.preconditioner,
                "filters": self.filters,
                "reg_weight": self.reg_weight,
                "reg_weight_update_strategy": self.reg_weight_update_strategy,
                "gradient_sclaer_config": self.gradient_scaler_config,
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

    def is_adaptive_regularization(self) -> bool:
        """Whether an adaptive regularization strategy is used."""
        if len(self.regularizators) == 0:
            return False
        return self.reg_weight_update_strategy.is_adaptive()

    def update(self, other: AdjustableParameter) -> None:
        """Update the attributes with other's."""
        for attr in other.__slots__:
            val = other.__getattribute__(attr)
            if val is not None:
                setattr(self, attr, val)

    def get_values_from_model_field(self, input_field: NDArrayFloat) -> None:
        """Update the parameter values from the given input field."""
        self.values = input_field

    def update_values_with_vector(
        self, update_values: NDArrayFloat, is_preconditioned: bool = False
    ) -> None:
        """Update the values attribute from the given vector.

        Parameters
        ----------
        update_values : NDArrayFloat
            The values used for update (preconditioned) in 1D.
        """
        if is_preconditioned:
            func: Callable = self.preconditioner.backtransform
        else:
            func = identify_function
        self.values = (
            func(update_values.ravel("F"))
            .clip(min=self.lbounds, max=self.ubounds)
            .reshape(self.values.shape, order="F")
        )

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
        n_vals = self.values.size
        if np.size(self.lbounds) == np.size(self.ubounds) == 1:
            bounds = np.concatenate(  # type: ignore
                [
                    np.full((1, n_vals), float(self.lbounds)),  # type: ignore
                    np.full((1, n_vals), float(self.ubounds)),  # type: ignore
                ]
            ).T
        else:
            bounds = np.array([self.lbounds, self.ubounds]).T

        # check that it has the correct shape
        assert bounds.shape == (n_vals, 2)

        # transform the bounds through the preconditioner
        if is_preconditioned:
            bounds = self.preconditioner.transform_bounds(bounds)
        return bounds

    def eval_loss_reg(self, s_raw: Optional[NDArrayFloat] = None) -> float:
        """
        Return the regularization objective function for the parameter.

        Parameters
        ----------
        s_raw: Optional[NDArrayFloat]
            Optional values for which to compute the regularization. If no values are
            provided, the values stored in the parameter instances are used.
            The values are not preconditioned.

        Note
        ----
        A ratio is applied.
        """
        _sum: float = 0.0
        for reg in self.regularizators:
            values: NDArrayFloat = self.values.copy()
            if s_raw is not None:
                values = s_raw
            _sum += reg.eval_loss(values.ravel("F"))
        return _sum

    def eval_loss_reg_gradient(
        self, s_raw: Optional[NDArrayFloat] = None
    ) -> NDArrayFloat:
        """
        Return the regularization objective function gradient as a 1D array.

        Parameters
        ----------
        s_raw: Optional[NDArrayFloat]
            Optional values for which to compute the regularization. If no values are
            provided, the values stored in the parameter instances are used.
            The values are not preconditioned.

        The returned gradient is not preconditioned.
        """
        grad: NDArrayFloat = np.zeros(self.values.size)
        for reg in self.regularizators:
            values: NDArrayFloat = self.values.copy()
            if s_raw is not None:
                values = s_raw
            # Note: I could add # .reshape(values.shape, order="F") to the next line
            # but I consider that the regularization must return a field of the
            # same shape as values. This is where the responsibility is.
            grad += reg.eval_loss_gradient(values.ravel("F"))
        return grad

    def save_reg_status(self, reg_param: float, loss_reg: float) -> None:
        """Save both the regularization weight and objective function."""
        # The loss_reg should be without the reg_param factor
        self.loss_reg_history.append(copy.copy(loss_reg))
        self.reg_weight_history.append(copy.copy(reg_param))

    def update_reg_weight(
        self,
        loss_ls_history: List[float],
        loss_ls_grad: NDArrayFloat,
        loss_reg_grad: NDArrayFloat,
        n_obs: int,
        logger: Optional[logging.Logger] = None,
    ) -> bool:
        """
        Update the regularization weight.

        Parameters
        ----------
        loss_ls_history : List[float]
            List of past LS cost function.
        loss_ls_grad : NDArrayFloat
            Current LS cost function gradient.
        loss_reg_grad : NDArrayFloat
            Current Reg cost function gradient.
        n_obs: int
            Number of observation which is an approximation of the noise level in the
            LS solution.
        logger: Optional[logging.Logger]
            Optional :class:`logging.Logger` instance used for event logging.
            The default is None.
        Returns
        -------
        bool
            Whether the regularization parameter (weight) has changed.
        """
        return self.reg_weight_update_strategy.update_reg_weight(
            loss_ls_history,
            self.loss_reg_history,
            self.reg_weight_history,
            loss_ls_grad,
            loss_reg_grad,
            n_obs,
            logger,
        )

    def get_values_change(
        self, is_use_pcd: bool = True, ord: Optional[float] = None
    ) -> float:
        """
        Evaluate the change between the two last vectors of values.

        The log of values is used for non additive properties such as permeability,
        storage coefficients or diffusion coefficient.
        """
        if len(self.archived_values) < 2:
            return 0

        def _pcd(x: NDArrayFloat) -> NDArrayFloat:
            # if self.is_scale_logarithmically:
            #     return np.log(x)
            if is_use_pcd:
                return self.preconditioner(x.ravel("F"))
            return x

        return sp.linalg.norm(
            _pcd(self.values) - _pcd(self.archived_values[-2]), ord=ord
        ) / sp.linalg.norm(_pcd(self.archived_values[-2]), ord=ord)


AdjustableParameters = Union[AdjustableParameter, Sequence[AdjustableParameter]]


def get_parameter_values_from_model(
    model: ForwardModel, param: AdjustableParameter
) -> NDArrayFloat:
    """
    Get the adjusted parameter values in the model.

    Note
    ----
    Returned values are not preconditioned.

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
    arr = get_array_from_state_variable(model, PARAM_TO_STATE_VAR[param.name], param.sp)
    if len(arr.shape) == 3:  # porosity, diffusion, perm, etc.
        return arr
    return arr[:, :, :, 0]  # initial head, pressure, concentrations and grade


def update_parameters_from_model(
    model: ForwardModel,
    parameters_to_adjust: AdjustableParameters,
) -> None:
    """
    Update adjusted parameters from model.

    Parameters
    ----------
    model: ForwardModel
        The forward model instance.
    parameters_to_adjust : AdjustableParameters
        Sequence of adjusted parameter instances.
    """
    for param in object_or_object_sequence_to_list(parameters_to_adjust):
        # check if the values are empty
        if param.values.size == 0:
            param.values = get_parameter_values_from_model(model, param)


def get_parameters_values_from_model(
    model: ForwardModel,
    params: AdjustableParameters,
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
        if is_preconditioned:
            data.append(
                param.preconditioner(
                    get_parameter_values_from_model(model, param).ravel("F")
                )
            )
        else:
            data.append(get_parameter_values_from_model(model, param).ravel("F"))
    # Concatenate the arrays and make it 1D
    return np.vstack(data).ravel()


def update_model_with_parameters_values(
    model: ForwardModel,
    parameters_values: NDArrayFloat,
    params: AdjustableParameters,
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
        update_model_with_param_values(model, param, param.sp)
        first_index = last_index
        # Store the values
        if is_to_save:
            param.archived_values.append(param.values.copy())


def get_parameters_bounds(
    params: AdjustableParameters,
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
    return param.preconditioner.dbacktransform_vec(
        param.preconditioner(param.values.ravel("F")),
        param.grad_adj_history[index].ravel("F"),
    ).reshape(param.values.shape, order="F")


def get_backconditioned_fd_gradient(
    param: AdjustableParameter, index: int
) -> NDArrayFloat:
    return param.preconditioner.dbacktransform_vec(
        param.preconditioner(param.values.ravel("F")),
        param.grad_fd_history[index].ravel("F"),
    ).reshape(param.values.shape, order="F")


def get_gridded_archived_gradients(
    param, is_adjoint: bool, is_preconditioned: bool = False
) -> NDArrayFloat:
    """Return an array (nx, ny, nt) of gridded archived gradients.

    Note
    ----
    The non adjusted grid cells are given as NaNs.

    Parameters
    ----------
    is_adjoint : bool
        Whether to use adjoint or finite differences gradients.

    Returns
    -------
    NDArrayFloat
        Array (nx, ny, nt) of gridded archived gradients.
    """
    if is_adjoint:
        if is_preconditioned:
            gradients: List[NDArrayFloat] = param.grad_adj_history
        else:
            gradients = param.grad_adj_raw_history
    else:
        gradients = param.grad_fd_history
    if gradients[0].size != param.values.size:
        raise ValueError("Impossible to reshape to grid dims!")
    out: NDArrayFloat = np.empty((*param.values.shape, len(gradients)))
    out[:] = np.nan
    for i, vals in enumerate(gradients):
        out[:, :, :, i] = vals.reshape(*param.values.shape, order="F")
    return out


def get_param_values(
    param: AdjustableParameter,
    is_preconditioned: bool = False,
) -> NDArrayFloat:
    """Return a slice of the input data field."""
    _values = param.values
    if is_preconditioned:
        return param.preconditioner(_values.ravel("F"))
    return _values


def update_model_with_param_values(
    model: ForwardModel, param: AdjustableParameter, sp: Optional[int] = None
) -> None:
    """Update the input field with the Adjustable parameter current values."""
    if param.name == ParameterName.INITIAL_CONCENTRATION:
        if sp is None:
            raise ValueError("sp cannot be None for concentrations!")
        model.tr_model.set_initial_conc(param.values, sp)
    elif param.name == ParameterName.INITIAL_HEAD:
        model.fl_model.set_initial_head(param.values)
    elif param.name == ParameterName.INITIAL_PRESSURE:
        model.fl_model.set_initial_pressure(param.values)
    elif param.name == ParameterName.INITIAL_GRADE:
        if sp is None:
            raise ValueError("sp cannot be None for grades!")
        model.tr_model.set_initial_grade(param.values, sp)
    elif param.name == ParameterName.PERMEABILITY:
        model.fl_model.permeability = param.values
    elif param.name == ParameterName.POROSITY:
        model.tr_model.porosity = param.values
    elif param.name == ParameterName.DIFFUSION:
        model.tr_model.diffusion = param.values
    elif param.name == ParameterName.DISPERSIVITY:
        model.tr_model.dispersivity = param.values
    elif param.name == ParameterName.STORAGE_COEFFICIENT:
        model.fl_model.storage_coefficient = param.values
    else:
        raise ValueError(
            f'"{param.name}" is not a valid state variable or parameter type!'
        )


def eval_weighted_loss_reg(
    params: AdjustableParameters,
    model: ForwardModel,
    s_raw: Optional[NDArrayFloat] = None,
    s_cond: Optional[NDArrayFloat] = None,
    is_save_reg_state: bool = False,
) -> float:
    """
    Get the regularization loss function for the provided parameters.

    Parameters
    ----------
    param : AdjustableParameters
        Sequence of adjutable parameters.
    model: ForwardModel
        Forward model from which the data are exctrated.
    s_raw : Optional[NDArrayFloat], optional
        Optional non-preconditioned values for the parameters.
        If None, the value stored in the parameter instances are used, by default None.
    s_cond : Optional[NDArrayFloat], optional
        Optional transformed values for the parameters. If None, the value stored in the
        parameter instances are used, by default None.

    Returns
    -------
    float
        The regularization objective function.
    """
    total_loss_reg: float = 0
    idx = 0
    for param in object_or_object_sequence_to_list(params):
        # not preconditioned values from model
        values: NDArrayFloat = get_parameter_values_from_model(model, param)
        # case 1 :not preconditoned values from input
        if s_raw is not None:
            values = s_raw[idx : idx + values.size]
            idx += values.size
        # case 2 : preconditoned values from input
        elif s_cond is not None:
            values = param.preconditioner.backtransform(s_cond[idx : idx + values.size])
            idx += values.size
        param_loss_reg: float = param.eval_loss_reg(values)
        if is_save_reg_state:
            param.save_reg_status(param.reg_weight, param_loss_reg)
        total_loss_reg += param_loss_reg * param.reg_weight
    return total_loss_reg


def eval_weighted_loss_reg_gradient(
    params: AdjustableParameters, model: ForwardModel, x: Optional[NDArrayFloat] = None
) -> NDArrayFloat:
    """
    Get the regularization loss function for the provided parameters.

    Parameters
    ----------
    param : AdjustableParameters
        Sequence of adjutable parameters.
    x : Optional[NDArrayFloat], optional
        Optional values for the parameters. If None, the value stored in the
        parameter instances are used, by default None. MUST BE PRECONDITONED !!!

    Returns
    -------
    The preconditoned gradient of regularization.
    """
    full_grad: NDArrayFloat = np.array([], dtype=np.float64)
    idx = 0
    for param in object_or_object_sequence_to_list(params):
        values: NDArrayFloat = get_parameter_values_from_model(model, param)
        if x is not None:
            values = x[idx : idx + values.size]
            idx += values.size
        # values must be unpreconditioned
        param_grad = param.eval_loss_reg_gradient(values) * param.reg_weight

        full_grad = np.hstack(
            (
                full_grad,
                param_grad,
            )
        )

    return full_grad
