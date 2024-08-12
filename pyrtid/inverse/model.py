"""Provide a model class to store the inversion parameters and results."""

from typing import List, Union

from pyrtid.forward.models import ForwardModel
from pyrtid.inverse.obs import Observable, Observables
from pyrtid.inverse.params import (
    AdjustableParameter,
    AdjustableParameters,
    ParameterName,
)
from pyrtid.utils.types import NDArrayFloat, object_or_object_sequence_to_list


class InverseModel:
    """
    Class holding the inversion parameters.

    Attributes
    ----------
    scaling_factor: float
        Multiplication coefficient to make the objective function equal to zero at
        the first iteration. In practise, solvers manage correctly in the range
        [-1e6, 1e6].
    is_check_gradient: bool
        Whether to check the gradient by finite difference at each iteration. This is
        expensive but very useful. If the gradient is not correct, an exception
        is raised.
    is_use_adjoint: bool
        Whether to use the adjoint for the gradient calculation.
    loss_scaled_history: List[float]
        List of successive objective functions computed while optimizing.
        Note: the values are not scaled.
    list_losses_for_fd_grad: List[float]
        List of successive objective functions computed while computing the gradients
        by finite difference. Note: the values are not scaled.
    list_d_pred: List[NDArrayFloat]
        List of the successive predicted vectors computed while optimizing.
    adj_conc: np.ndarray
        Adjoint concentrations.
    parameters_to_adjust: List[AdjustableParameter]
        List of adjustable parameters considered in the inversion.
    is_first_loss_function_call_in_round: bool
        Whether it is the first optimization round (minimize call) or not.
    optimization_round_nb: int
        Nb of times the optimization routine (minimize) has been called.
    is_regularization_at_first_round: bool, optional
        Set the regularization on the first optimization round. The default is True.

    """

    __slots__ = [
        "observables",
        "parameters_to_adjust",
        "optimization_round_nb",
        "scaling_factor",
        "is_regularization_at_first_round",
        "nb_g_calls",
        "loss_ls_unscaled",
        "loss_reg_unscaled",
        "loss_ls_history",
        "loss_reg_weighted_history",
        "loss_scaled_history",
        "list_losses_for_fd_grad",
        "list_d_pred",
        "is_first_loss_function_call_in_round",
        "n_update_rw",
    ]

    def __init__(
        self,
        parameters_to_adjust: AdjustableParameters,
        observables: Observables,
    ) -> None:
        """
        Initialize the instance.

        Parameters
        ----------
        parameters_to_adjust : AdjustableParameters
            _description_
        observables : Observables
            _description_

        Raises
        ------
        ValueError
            _description_
        ValueError
            _description_
        """

        self.observables: List[Observable] = object_or_object_sequence_to_list(
            observables
        )
        self.parameters_to_adjust: List[AdjustableParameter] = (
            object_or_object_sequence_to_list(parameters_to_adjust)
        )

        if len(self.parameters_to_adjust) == 0:
            raise ValueError(
                "You must provide at least one AdjustableParameter instance!"
            )
        if len(self.observables) == 0:
            raise ValueError("You must provide at least one Observable instance!")

        # Initialize the internal state (lists, comptors, etc.)
        self.init_state()

    def init_state(self) -> None:
        """Initialize internal state."""

        self.optimization_round_nb: int = 0
        self.scaling_factor: float = 1.0
        self.is_regularization_at_first_round: bool = True
        self.nb_g_calls = 0
        self.loss_ls_history: List[float] = []
        self.loss_reg_weighted_history: List[float] = []
        self.list_losses_for_fd_grad: List[float] = []
        self.loss_scaled_history: List[float] = []
        self.list_d_pred: List[NDArrayFloat] = []
        self.is_first_loss_function_call_in_round: bool = True
        self.loss_ls_unscaled: float = 0.0
        self.loss_reg_unscaled: float = 0.0
        self.n_update_rw: int = 0

    @property
    def nb_f_calls(self) -> int:
        """Return the number of times the objective function has been called."""
        return len(self.loss_scaled_history) + len(self.list_losses_for_fd_grad)

    @property
    def nb_adjusted_values(self) -> int:
        """Return the number of adjusted values in the inversion."""
        return sum(
            [
                int(param.size_preconditioned_values)
                for param in self.parameters_to_adjust
            ]
        )

    @property
    def nb_obs_values(self) -> int:
        """Return the number of observation values in the inversion."""
        return sum([int(obs.values.size) for obs in self.observables])

    @property
    def loss_total_unscaled(self) -> float:
        """Return the total loss (unscaled)."""
        return self.loss_ls_unscaled + self.loss_reg_unscaled

    def is_new_optimization_round_needed(self, max_optimization_round_nb: int) -> bool:
        """
        Return whether the optimization loop needs to restart or not.

        The goal of the restart is mainly to recompute the weights between the
        residuals objective function and the regularization terms.

        Parameters
        ----------
        max_optimization_round_nb: int
            The maximum number of optimization rounds.

        """
        # Case of first loop
        if self.optimization_round_nb < 1:
            return True
        # Case without regularization
        if all([len(param.regularizators) == 0 for param in self.parameters_to_adjust]):
            return False
        # Stop criteria
        if self.optimization_round_nb < max_optimization_round_nb:
            return True
        return False

    def is_adaptive_regularization(self) -> bool:
        """Whether any adjusted parameter use adaptive regularization strategy."""
        for param in self.parameters_to_adjust:
            if param.is_adaptive_regularization():
                return True
        return False

    def get_loss_function_scaling_factor(
        self, loss_function: float, is_first_call: bool
    ) -> float:
        """Return the scaling factor so the loss function is 1.0 at first evaluation."""
        # First iteration -> we compute a coefficient so that J ~ 1.
        if is_first_call:
            if loss_function == 0.0:
                self.scaling_factor = 1.0
            else:
                self.scaling_factor = 1.0 / loss_function
        return self.scaling_factor

    def set_parameters_to_adjust(
        self,
        parameters_to_adjust: AdjustableParameters,
        model: ForwardModel,
    ) -> None:
        """Set the parameters to adjust during the inversion."""
        if isinstance(parameters_to_adjust, list):
            self.parameters_to_adjust = parameters_to_adjust
        elif isinstance(parameters_to_adjust, AdjustableParameter):
            self.parameters_to_adjust = [parameters_to_adjust]
        else:
            raise ValueError(
                "'parameters_to_adjust' should be of type Sequence[AdjustableParameter]"
            )

        # Initiate the parameters values from the model + spatial regularization
        for param in self.parameters_to_adjust:
            if param.name not in ParameterName.to_list():
                raise ValueError(
                    f"{param.name} is not an adjustable parameter !\n"
                    f"Supported parameters are {ParameterName.to_list()}"
                )
            if param.name == ParameterName.DIFFUSION:
                param.get_values_from_model_field(model.tr_model.diffusion)
            elif param.name == ParameterName.POROSITY:
                param.get_values_from_model_field(model.tr_model.porosity)
            elif param.name == ParameterName.PERMEABILITY:
                param.get_values_from_model_field(model.fl_model.permeability)
            elif param.name == ParameterName.INITIAL_CONCENTRATION:
                param.get_values_from_model_field(model.tr_model.mob[:, :, 0])
            elif param.name == ParameterName.INITIAL_GRADE:
                param.get_values_from_model_field(model.tr_model.immob[:, :, 0])
            else:
                raise (
                    NotImplementedError(
                        "Please contact the developer to handle this issue."
                    )
                )

    def set_observables(self, observables: Union[Observable, List[Observable]]) -> None:
        """Set the parameters to adjust during the inversion."""
        if isinstance(observables, list):
            self.observables = observables
        elif isinstance(observables, Observable):
            self.observables = [observables]
        else:
            raise ValueError("'observable' should be of type List[Observable]")

    def clear_history(self) -> None:
        """Clear the history."""
        self.init_state()
        # clear param history as well
        for param in self.parameters_to_adjust:
            param.init_state()
