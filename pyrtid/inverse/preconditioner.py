"""
Provides preconditioners for the adjusted (updated values).

Classes
=======

Abstract interface
^^^^^^^^^^^^^^^^^^

Abstract interface. For linting or to use as base to create custom preconditioners.

.. autosummary::
   :toctree: _autosummary

    Preconditioner

Preconditioners/Transformers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Preconditioners for users. :class:`NoTransform` does not apply any changes.
They can be combined through the :class:`ChainedTransforms` interface.

.. autosummary::
   :toctree: _autosummary

    NoTransform
    LinearTransform
    SqrtTransform
    InvAbsTransform
    LogTransform
    SigmoidRescaler
    Normalizer
    StdRescaler
    BoundsRescaler
    GDPCS
    GDPNCS
    ChainedTransforms
    SubSelector
    Slicer
    Uniform2Gaussian
    BoundsClipper

Gradient Scaling
^^^^^^^^^^^^^^^^

Configuration for the gradient scaling approach with L-BFGS-B.

.. autosummary::
   :toctree: _autosummary

    GradientScalerConfig


Functions
=========

Forward
^^^^^^^

Transformation functions used in preconditioners.

.. autosummary::
   :toctree: _autosummary

    logistic
    logit
    tanh_wrapper
    arctanh_wrapper
    to_new_range
    get_gd_weights
    get_theta_init
    get_theta_init_uniform
    get_theta_init_normal
    gd_parametrize

Derivative
^^^^^^^^^^

Transformation functions derivatives.

.. autosummary::
   :toctree: _autosummary

    dtanh_wrapper
    darctanh_wrapper
    to_new_range_derivative
    d_gd_parametrize_mat_vec

"""

import copy
import logging
import pickle
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Generator, List, Optional, Sequence, Tuple, Union

import numdifftools as nd
import numpy as np
import scipy as sp
from scipy._lib._util import check_random_state  # To handle random_state
from scipy.sparse import csc_array, lil_array
from sksparse.cholmod import Factor

import pyrtid.utils.spde as spde
from pyrtid.forward.models import Geometry
from pyrtid.utils import object_or_object_sequence_to_list, sparse_cholesky
from pyrtid.utils.types import NDArrayBool, NDArrayFloat, NDArrayInt


class Preconditioner(ABC):
    """
    This an asbract class for parameter preconditioning and parametrization.

    This class provides an interface for adjusted variables preconditioning i.e.,
    application of a transformation, that conditions a given problem into a form that
    is more suitable for numerical solving methods. The interface is the same for
    parametrization, i.e. reduction of the number of adjusted values.
    """

    # These bounds are used to ensure that the transform and the associated back-
    # transform are defined for the given values (s_raw and s_cond)
    # These are to be redefined (hardcoded) in derived classes if different
    LBOUND_RAW: float = -np.inf
    UBOUND_RAW: float = +np.inf
    LBOUND_COND: float = -np.inf
    UBOUND_COND: float = +np.inf

    def __init__(self) -> None:
        """Initialize the instance."""

    def transform(self, s_raw: NDArrayFloat) -> NDArrayFloat:
        """
        Apply the preconditioning/parametrization.

        Parameters
        ----------
        s_raw : NDArrayFloat
            The non-conditioned values as a 1D array.

        Returns
        -------
        NDArrayFloat
            The conditioned values as a 1D vector.
        """
        if not s_raw.ndim == 1:
            raise ValueError("'transfrom' method expects a 1D vector!")
        self.test_bounds_tr(s_raw)  # test that s_raw is in the supported range
        # call the _transform method defined in child classes
        return self._transform(s_raw)

    def backtransform(self, s_cond: NDArrayFloat) -> NDArrayFloat:
        """
        Apply the back-preconditioning/parametrization.

        Parameters
        ----------
        s_raw : NDArrayFloat
            The conditioned values as a 1D array.

        Returns
        -------
        NDArrayFloat
            The non-conditioned values as a 1D vector.
        """
        if not s_cond.ndim == 1:
            raise ValueError("'backtransfrom' method expects a 1D vector!")
        self.test_bounds_btr(s_cond)  # test that s_cond is in the supported range
        # call the _backtransform method defined in child classes
        return self._backtransform(s_cond)

    @abstractmethod
    def _transform(self, s_raw: NDArrayFloat) -> NDArrayFloat:
        """Apply the preconditioning/parametrization."""
        ...  # pragma: no cover

    @abstractmethod
    def _backtransform(self, s_cond: NDArrayFloat) -> NDArrayFloat:
        """Apply the back-preconditioning/parametrization."""
        ...  # pragma: no cover

    def dtransform_vec(
        self, s_raw: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat:
        """
        Return the transform 1st derivative times a vector as a 1-D vector..

        Because the preconditioner operates a variable change in the function: the new
        objective function J is J2(s2) = J[s], with s the adjusted parameter vector.
        Then the gradient is dJ/ds = ds2[s]/ds * dJ2[s2]/ds2. This is useful when the
        derivative is computed on s2 but is required w.r.t s.

        - Example 1 : we defined s2[s] = k * s
          -> dJ/ds = ds2[s]/ds dJ2[s2]/ds2 = k * dJ2/ds2. If
          k = 1/100 --> the gradient dJ[s]/ds is a 100 times weaker than dJ2/ds2.
        - Example 2: we defined s2 = log(s)
          -> dJ/ds = ds2[s]/ds dJ2[s2]/ds2 = 1/s * dJ2[s2]/ds2

        Often, it is more efficient to compute (ds2/ds * dJ2/ds2) directly
        than to return ds2/ds, espectially if ds2/ds is a matrix of large dimension.

        Parameters
        ----------
        b : NDArrayFloat
            Any vector with size $N_{s}$.

        Returns
        -------
        NDArrayFloat
            Product of the 1st derivative w.r.t. the conditioned (transformed)
            values and any vector b with size $N_{s2}$.
        """
        if not s_raw.ndim == 1 or not gradient.ndim == 1:
            raise ValueError("'dtransfrom_vec' method expects 1D vectors!")
        self.test_bounds_tr(s_raw)  # test that s_raw is in the supported range
        # call the _dtransform_vec method defined in child classes
        return self._dtransform_vec(s_raw, gradient)

    @abstractmethod
    def _dtransform_vec(
        self, s_raw: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat: ...  # pragma: no cover

    def dbacktransform_vec(
        self, s_cond: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat:
        """
        Return the backtransform 1st derivative times a vector.

        Because the preconditioner operates a variable change in the function: the new
        objective function J is J2(s2) = J[s], with s the adjusted parameter vector.
        Then the gradient is dJ2/ds2 = ds/ds2 * dJ/ds. This is useful when the
        derivative is computed on s but is required w.r.t s2.

        - Example 1 : we defined s[s2] = s2 / k (backtransform operation)
          -> dJ2/ds2 = ds[s2]/ds2 dJ/ds = 1/k * dJ/ds.
          If k = 1/100 --> the gradient dJ2/ds2 is a 100 times stronger than dJ/ds.
          The parameter update is performed on s2 and not on s.

        - Example 2: we defined s[s2] = exp(s2)
          -> dJ2/ds2 = ds/ds2 dJ/ds = exp(s2) * dj/ds

        Often, it is more efficient to compute (ds/ds2 * dJ/ds) directly
        than to return ds/ds2, espectially if ds/ds2 is a matrix of large dimension.

        Parameters
        ----------
        b : NDArrayFloat
            Any vector with size $N_{s}$.

        Returns
        -------
        NDArrayFloat
            Product of the 1st derivative w.r.t. the conditioned (transformed)
            values and any vector b with size $N_{s}$.
        """
        if not s_cond.ndim == 1 or not gradient.ndim == 1:
            raise ValueError("'dtransfrom_vec' method expects 1D vectors!")

        self.test_bounds_btr(s_cond)  # test that s_cond is in the supported range
        # call the _dbacktransform_vec method defined in child classes
        return self._dbacktransform_vec(s_cond, gradient)

    def dbacktransform_inv_vec(
        self, s_cond: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat:
        """
        Return the inverse of the backtransform 1st derivative times a vector.

        Because the preconditioner operates a variable change in the function: the new
        objective function J is J2(s2) = J[s], with s the adjusted parameter vector.
        Then the gradient is dJ2/ds2 = ds/ds2 * dJ/ds. This is useful when the
        derivative is computed on s but is required w.r.t s2.

        - Example 1 : we defined s[s2] = s2 / k (backtransform operation)
          -> dJ2/ds2 = ds[s2]/ds2 dJ/ds = 1/k * dJ/ds.
          If k = 1/100 --> the gradient dJ2/ds2 is a 100 times stronger than dJ/ds.
          The parameter update is performed on s2 and not on s.

        - Example 2: we defined s[s2] = exp(s2)
          -> dJ2/ds2 = ds/ds2 dJ/ds = exp(s2) * dj/ds

        Often, it is more efficient to compute (ds/ds2 * dJ/ds) directly
        than to return ds/ds2, espectially if ds/ds2 is a matrix of large dimension.

        Parameters
        ----------
        b : NDArrayFloat
            Any vector with size $N_{s}$.

        Returns
        -------
        NDArrayFloat
            Product of the 1st derivative w.r.t. the conditioned (transformed)
            values and any vector b with size $N_{s}$.
        """
        if not s_cond.ndim == 1 or not gradient.ndim == 1:
            raise ValueError("'dbacktransfrom_inv_vec' method expects 1D vectors!")

        self.test_bounds_btr(s_cond)  # test that s_cond is in the supported range
        # call the _dbacktransform_vec method defined in child classes
        return self._dbacktransform_inv_vec(s_cond, gradient)

    @abstractmethod
    def _dbacktransform_vec(
        self, s_cond: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat: ...  # pragma: no cover

    @abstractmethod
    def _dbacktransform_inv_vec(
        self, s_cond: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat: ...  # pragma: no cover

    def __call__(self, s_raw: NDArrayFloat) -> NDArrayFloat:
        """Call the preconditioner."""
        return self.transform(s_raw)

    def _get_test_data(
        self,
        lbounds: Union[float, NDArrayFloat],
        ubounds: Union[float, NDArrayFloat],
        shape: Optional[Union[int, Sequence[int]]] = None,
    ) -> NDArrayFloat:
        """
        Get test data to check the preconditioner correctness.

        This is a development tool.
        """
        _lbounds, _ubounds = np.array(lbounds), np.array(ubounds)
        _lbounds[np.isneginf(_lbounds)] = -1e10
        _ubounds[np.isposinf(_ubounds)] = +1e10
        if shape is None:
            _size = np.size(lbounds)
            if _size == 1:
                _size = (50,)
        else:
            _size = shape
        # uniform sampling
        test_data: NDArrayFloat = np.random.default_rng(2023).uniform(
            low=_lbounds, high=_ubounds, size=_size
        )
        return test_data

    def test_bounds_tr(self, s_raw: NDArrayFloat) -> None:
        """
        Test the bounds for transformation.

        Parameters
        ----------
        s_raw : NDArrayFloat
            NOn-conditioned values.

        Raises
        ------
        ValueError
            If the provided values are out of the supported range.
        """
        if np.any(np.logical_or(self.LBOUND_RAW > s_raw, self.UBOUND_RAW < s_raw)):
            raise ValueError(
                "The provided parameter bounds "
                f"[{np.min(s_raw)}, {np.max(s_raw)}] "
                "do not match with the "
                "range of values supported by the transform: "
                f"[{self.LBOUND_RAW}, {self.UBOUND_RAW}]"
            )

    def test_bounds_btr(self, s_cond: NDArrayFloat) -> None:
        """

        Parameters
        ----------
        s_cond: NDArrayFloat
            Conditioned values to back-transform.

        Raises
        ------
        ValueError
            If the provided values are out of the supported range.
        """

        if np.any(np.logical_or(self.LBOUND_COND > s_cond, self.UBOUND_COND < s_cond)):
            raise ValueError(
                "The provided parameter bounds "
                f"[{np.min(s_cond)}, {np.max(s_cond)}] "
                "do not match with the "
                "range of values supported by the back-transform: "
                f"[{self.LBOUND_COND}, {self.UBOUND_COND}]"
            )

    def test_preconditioner(
        self,
        lbounds: Union[float, NDArrayFloat],
        ubounds: Union[float, NDArrayFloat],
        shape: Optional[Union[int, Sequence[int]]] = None,
        rtol: float = 1e-5,
        eps: Optional[float] = None,
    ) -> None:
        """
        Test if the backconditioner and the derivatives times a vector are correct.

        This is a development tool.

        Parameters
        ----------
        lbounds : Union[float, NDArrayFloat]
            _description_
        ubounds : Union[float, NDArrayFloat]
            _description_
        shape : Optional[Union[int, Sequence[int]]], optional
            _description_, by default None
        rtol : float, optional
            _description_, by default 1e-5
        eps : Optional[float], optional
            The epsilon for the computation of the approximated preconditioner first
            derivative by finite difference. by default None.

        Raises
        ------
        ValueError
            If one of the backconditioner of the gradient conditioner are incorrect.
        """
        self._test_preconditioner(lbounds, ubounds, shape, rtol, eps)

    def _test_preconditioner(
        self,
        lbounds: Union[float, NDArrayFloat],
        ubounds: Union[float, NDArrayFloat],
        shape: Optional[Union[int, Sequence[int]]] = None,
        rtol: float = 1e-5,
        eps: Optional[float] = None,
        skip_checks: Optional[Sequence[int]] = None,
    ) -> None:
        """
        Test if the backconditioner and the derivatives times a vector are correct.

        This is a development tool.

        Parameters
        ----------
        lbounds : Union[float, NDArrayFloat]
            _description_
        ubounds : Union[float, NDArrayFloat]
            _description_
        shape : Optional[Union[int, Sequence[int]]], optional
            _description_, by default None
        rtol : float, optional
            _description_, by default 1e-5
        eps : Optional[float], optional
            The epsilon for the computation of the approximated preconditioner first
            derivative by finite difference. by default None.
        skip_checks: Optional[Sequence[int]]
            List of checks to skip. This is useful when some preconditioner will fail
            tests while remaining correct. The default is None.

        Raises
        ------
        ValueError
            If one of the backconditioner of the gradient conditioner are incorrect.
        """
        # Add a small epsilon to avoid boundary cases
        test_data = self._get_test_data(lbounds=lbounds, ubounds=ubounds, shape=shape)

        _skip_checks = skip_checks if skip_checks is not None else []

        # 1) check if the back and pre-conditioner match
        if 1 not in _skip_checks:
            try:
                np.testing.assert_allclose(
                    test_data, self.backtransform(self.transform(test_data)), rtol=rtol
                )
            except AssertionError as e:
                raise ValueError(
                    "The given backconditioner does not match the preconditioner! or"
                    " the provided bounds are not correct."
                ) from e

        # 2) check by finite difference if the pre-conditioner 1st derivative is correct
        # transform to ensure the correct size
        if 2 not in _skip_checks:
            gradient = self.transform(test_data)
            np.testing.assert_allclose(
                self.dtransform_vec(test_data, gradient),
                # Finite difference differentiation
                nd.Jacobian(self.transform, step=eps)(test_data).T @ gradient,
                rtol=rtol,
            )  # type: ignore

        # 3) check by finite difference if the back-conditioner derivative is correct
        if 3 not in _skip_checks:
            gradient = test_data
            np.testing.assert_allclose(
                self.dbacktransform_vec(self.transform(test_data), gradient),
                # Finite difference differentiation
                nd.Jacobian(self.backtransform, step=eps)(self.transform(test_data)).T
                @ gradient,  # type: ignore
                rtol=rtol,
            )

        # 4) check that dbacktransform_inv_vec is correct
        # Note: for some preconditioners, this function does not exists. In that case
        # it expects a NotImplementedError
        if 4 not in _skip_checks:
            try:
                np.testing.assert_allclose(
                    self.dbacktransform_inv_vec(
                        self.transform(test_data),
                        self.dbacktransform_vec(self.transform(test_data), gradient),
                    ),
                    gradient,
                )
            except NotImplementedError:
                pass

    def transform_bounds(self, bounds: NDArrayFloat) -> NDArrayFloat:
        """
        Transform the bounds to match the preconditioned values.

        Parameters
        ----------
        bounds : NDArrayFloat
            Array of shape (N_s, 2).

        Returns
        -------
        NDArrayFloat
            Array of shape (N_s, 2) with transformed bounds.
        """
        # Apply the preconditioning to lower and upper bounds and sort the values
        # so that lbounds <= ubounds
        # this assumes that the underlying transformation is monotonic.
        # hence, this function must be modified in child class if this assumption
        # does not hold
        return np.sort(np.array([self(bounds[:, 0]), self(bounds[:, 1])]).T, axis=1)


class ChainedTransforms(Preconditioner):
    """Combinaison of multiple preconditioners."""

    def __init__(self, pcds: Sequence[Preconditioner]) -> None:
        """
        Initialize the instance.

        Parameters
        ----------
        pcds : Sequence[Preconditioner]
            Sequence of preconditioners to apply. The preconditioners are applied in
            the order of the given list.
        """
        self.pcds: List[Preconditioner] = object_or_object_sequence_to_list(pcds)

    def _transform(self, s_raw: NDArrayFloat) -> NDArrayFloat:
        """
        Apply the preconditioning/parametrization.

        It works a bit differently for this preconditioner -> the value is not taken
        from the model.

        Parameters
        ----------
        s_raw : NDArrayFloat
            Non-conditioned (transformed) parameter values.

        Returns
        -------
        NDArrayFloat
            Conditioned (transformed) parameter values.
        """

        def chain_op(input, i) -> NDArrayFloat:
            """Chain the preconditioners."""
            if i < len(self.pcds):
                return chain_op(self.pcds[i].transform(input), i + 1)
            return input

        # successively transform the data with each preconditioner
        return chain_op(s_raw, 0)

    def _backtransform(self, s_cond: NDArrayFloat) -> NDArrayFloat:
        """
        Apply the back-preconditioning/parametrization.

        Parameters
        ----------
        s_cond : NDArrayFloat
            Conditioned (transformed) parameter values.

        Returns
        -------
        NDArrayFloat
            Non-conditioned (transformed) parameter values.
        """

        def chain_back_op(input, i) -> NDArrayFloat:
            """Chain the preconditioners."""
            if i < len(self.pcds):
                return chain_back_op(
                    self.pcds[len(self.pcds) - 1 - i].backtransform(input), i + 1
                )
            return input

        # successively backtransform the data with each preconditioner
        # in the reversed order
        return chain_back_op(s_cond, 0)

    def _dtransform_vec(
        self, s_raw: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat:
        """
        Return the transform 1st derivative times a vector as a 1-D vector..
        """

        def _chain_derivative_op(s, _gradient, i) -> NDArrayFloat:
            """Chain the preconditioners."""
            if i < len(self.pcds):
                pcd: Preconditioner = self.pcds[i]
                return _chain_derivative_op(
                    pcd(s), pcd.dtransform_vec(s, _gradient), i + 1
                )
            return _gradient

        return _chain_derivative_op(s_raw, gradient, 0)

    def _dbacktransform_vec(
        self, s_cond: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat:
        """
        Return the backtransform 1st derivative times a vector.
        """

        # dJ3/ds3 = ds2[s3]/ds3 * ds[s2]/ds2 * dJ/ds ...
        def _chain_derivative_op(s, _gradient, i) -> NDArrayFloat:
            """Chain the preconditioners."""
            if i < len(self.pcds):
                pcd: Preconditioner = self.pcds[i]
                return _chain_derivative_op(
                    pcd(s), pcd.dbacktransform_vec(pcd(s), _gradient), i + 1
                )
            return _gradient

        return _chain_derivative_op(self.backtransform(s_cond), gradient, 0)

    def _dbacktransform_inv_vec(
        self, s_cond: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat:
        """
        Return the inverse of the backtransform 1st derivative times a vector.
        """

        def chain_derivative_back_inv_op(s, _gradient, i) -> NDArrayFloat:
            """Chain the preconditioners."""
            if i < len(self.pcds):
                pcd: Preconditioner = self.pcds[len(self.pcds) - 1 - i]
                return chain_derivative_back_inv_op(
                    pcd.backtransform(s),
                    pcd.dbacktransform_inv_vec(s, _gradient),
                    i + 1,
                )
            return _gradient

        # successively backtransform the data with each preconditioner
        # in the reversed order
        return chain_derivative_back_inv_op(s_cond, gradient, 0)

    def transform_bounds(self, bounds: NDArrayFloat) -> NDArrayFloat:
        """
        Transform the bounds to match the preconditioned values.

        Parameters
        ----------
        bounds : NDArrayFloat
            Array of shape (N_s, 2).

        Returns
        -------
        NDArrayFloat
            Array of shape (N_s, 2) with transformed bounds.
        """

        # Apply the preconditioning to lower and upper bounds and sort the values
        # so that lbounds <= ubounds
        # this assumes that the underlying transformation is monotonic.
        # hence, this function must be modified in child class if this assumption
        # does not hold
        def chain_op(input, i) -> NDArrayFloat:
            """Chain the preconditioners."""
            if i < len(self.pcds):
                return chain_op(self.pcds[i].transform_bounds(input), i + 1)
            return input

        # successively transform the data with each preconditioner
        return chain_op(bounds, 0)


class NoTransform(Preconditioner):
    """Does not apply any preconditioning."""

    def _transform(self, s_raw: NDArrayFloat) -> NDArrayFloat:
        """
        Apply the preconditioning/parametrization.

        Parameters
        ----------
        s_raw : NDArrayFloat
            Non-conditioned (transformed) parameter values.

        Returns
        -------
        NDArrayFloat
            Conditioned (transformed) parameter values.
        """
        return s_raw

    def _backtransform(self, s_cond: NDArrayFloat) -> NDArrayFloat:
        """
        Apply the back-preconditioning/parametrization.

        Parameters
        ----------
        s_cond : NDArrayFloat
            Conditioned (transformed) parameter values.

        Returns
        -------
        NDArrayFloat
            Non-Conditioned (transformed) parameter values.
        """
        return s_cond

    def _dtransform_vec(
        self, s_raw: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat:
        """
        Return the transform 1st derivative times a vector as a 1-D vector..
        """
        return gradient

    def _dbacktransform_vec(
        self, s_cond: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat:
        """
        Return the backtransform 1st derivative times a vector.
        """
        return gradient

    def _dbacktransform_inv_vec(
        self, s_cond: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat:
        """
        Return the inverse of the backtransform 1st derivative times a vector.
        """
        return gradient


class LinearTransform(Preconditioner):
    """Apply a linear transform to the parameter."""

    def __init__(
        self, slope: Union[float, NDArrayFloat], y_intercept: Union[float, NDArrayFloat]
    ) -> None:
        """
        Initialize the instance.

        Parameters
        ----------
        slope : float
            Slope of the transform.
        y_intercept : float
            Value when the parameter is null.
        """
        self.slope = slope
        self.y_intercept = y_intercept

    def _transform(self, s_raw: NDArrayFloat) -> NDArrayFloat:
        """
        Apply the preconditioning/parametrization.

        Parameters
        ----------
        s_raw : NDArrayFloat
            Non-conditioned (transformed) parameter values.

        Returns
        -------
        NDArrayFloat
            Conditioned (transformed) parameter values.
        """
        return s_raw * self.slope + self.y_intercept

    def _backtransform(self, s_cond: NDArrayFloat) -> NDArrayFloat:
        """
        Apply the back-preconditioning/parametrization.

        Parameters
        ----------
        s_cond : NDArrayFloat
            Conditioned (transformed) parameter values.

        Returns
        -------
        NDArrayFloat
            Non-conditioned (transformed) parameter values.
        """
        return (s_cond - self.y_intercept) / self.slope

    def _dtransform_vec(
        self, s_raw: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat:
        """
        Return the transform 1st derivative times a vector as a 1-D vector..
        """
        return self.slope * gradient

    def _dbacktransform_vec(
        self, s_cond: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat:
        """
        Return the backtransform 1st derivative times a vector.
        """
        return 1 / self.slope * gradient

    def _dbacktransform_inv_vec(
        self, s_cond: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat:
        """
        Return the inverse of the backtransform 1st derivative times a vector.
        """
        return gradient * self.slope


class SqrtTransform(Preconditioner):
    """Apply a sqrt preconditioning to ensure positive values of the parameter."""

    LBOUND_RAW: float = 0.0
    UBOUND_RAW: float = +np.inf

    def _transform(self, s_raw: NDArrayFloat) -> NDArrayFloat:
        """
        Apply the preconditioning/parametrization.

        Parameters
        ----------
        s_raw : NDArrayFloat
            Non-conditioned (transformed) parameter values.

        Returns
        -------
        NDArrayFloat
            Conditioned (transformed) parameter values.
        """
        return np.sqrt(s_raw)

    def _backtransform(self, s_cond: NDArrayFloat) -> NDArrayFloat:
        """
        Apply the back-preconditioning/parametrization.

        Parameters
        ----------
        s_cond : NDArrayFloat
            Conditioned (transformed) parameter values.

        Returns
        -------
        NDArrayFloat
            Non-Conditioned (transformed) parameter values.
        """
        return np.square(s_cond)

    def _dtransform_vec(
        self, s_raw: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat:
        """
        Return the transform 1st derivative times a vector as a 1-D vector..
        """
        return 1 / (2.0 * np.sqrt(s_raw)) * gradient

    def _dbacktransform_vec(
        self, s_cond: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat:
        """
        Return the backtransform 1st derivative times a vector.
        """
        return 2.0 * s_cond * gradient

    def _dbacktransform_inv_vec(
        self, s_cond: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat:
        """
        Return the inverse of the backtransform 1st derivative times a vector.
        """
        return gradient / 2.0 / s_cond


class InvAbsTransform(Preconditioner):
    """Apply an inverse absolute value preconditioning to ensure of the parameter."""

    LBOUND_RAW: float = 0.0
    UBOUND_RAW: float = +np.inf

    def __init__(self) -> None:
        self.signs = np.array([1.0])

    def _transform(self, s_raw: NDArrayFloat) -> NDArrayFloat:
        """
        Apply the preconditioning/parametrization.

        Parameters
        ----------
        s_raw : NDArrayFloat
            Non-conditioned (transformed) parameter values.

        Returns
        -------
        NDArrayFloat
            Conditioned (transformed) parameter values.
        """
        return self.signs * s_raw

    def _backtransform(self, s_cond: NDArrayFloat) -> NDArrayFloat:
        """
        Apply the back-preconditioning/parametrization.

        Parameters
        ----------
        s_cond : NDArrayFloat
            Conditioned (transformed) parameter values.

        Returns
        -------
        NDArrayFloat
            Non-conditioned (transformed) parameter values.
        """
        self.signs = np.sign(s_cond)
        return np.abs(s_cond)

    def _dtransform_vec(
        self, s_raw: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat:
        """
        Return the transform 1st derivative times a vector as a 1-D vector..
        """
        return self.signs * gradient

    def _dbacktransform_vec(
        self, s_cond: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat:
        """
        Return the backtransform 1st derivative times a vector.
        """
        return np.sign(s_cond) * gradient

    def _dbacktransform_inv_vec(
        self, s_cond: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat:
        """
        Return the inverse of the backtransform 1st derivative times a vector.
        """
        # should have the same effect as gradient / np.sign(s_sond)
        return gradient * np.sign(s_cond)

    def transform_bounds(self, bounds: NDArrayFloat) -> NDArrayFloat:
        """
        Transform the bounds to match the preconditioned values.

        Parameters
        ----------
        bounds : NDArrayFloat
            Array of shape (N_s, 2).

        Returns
        -------
        NDArrayFloat
            Array of shape (N_s, 2) with transformed bounds.
        """
        return bounds


class LogTransform(Preconditioner):
    """Apply a sqrt preconditioning to ensure positive values of the parameter."""

    LBOUND_RAW = 1e-30  # cannot be zero
    UBOUND_RAW = +np.inf

    def _transform(self, s_raw: NDArrayFloat) -> NDArrayFloat:
        """
        Apply the preconditioning/parametrization.

        Parameters
        ----------
        s_raw : NDArrayFloat
            Non-conditioned (transformed) parameter values.

        Returns
        -------
        NDArrayFloat
            Conditioned (transformed) parameter values.
        """
        return np.log(s_raw)

    def _backtransform(self, s_cond: NDArrayFloat) -> NDArrayFloat:
        """
        Apply the back-preconditioning/parametrization.

        Parameters
        ----------
        s_cond : NDArrayFloat
            Conditioned (transformed) parameter values.

        Returns
        -------
        NDArrayFloat
            Non-conditioned (transformed) parameter values.
        """
        return np.exp(s_cond)

    def _dtransform_vec(
        self, s_raw: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat:
        """
        Return the transform 1st derivative times a vector as a 1-D vector..
        """
        return 1 / s_raw * gradient

    def _dbacktransform_vec(
        self, s_cond: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat:
        """
        Return the backtransform 1st derivative times a vector.
        """
        return np.exp(s_cond) * gradient

    def _dbacktransform_inv_vec(
        self, s_cond: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat:
        """
        Return the inverse of the backtransform 1st derivative times a vector.
        """
        return gradient / np.exp(s_cond)


def logistic(
    s: NDArrayFloat, s0: float = 0.0, rate: float = 1.0, supremum: float = 1.0
) -> NDArrayFloat:
    """
    Return the logistic function (inverse to logit).

    Parameters
    ----------
    s : NDArrayFloat
        Input values.
    s0 : float, optional
        Value of the function's midpoint, by default 0.0?
    rate : float, optional
        The logistic growth rate or steepness of the curve, by default 1.0
    supremum : float, optional
        The supremum of the values of the function, by default 1.0

    Returns
    -------
    NDArrayFloat
        Logistic values.
    """
    return supremum / (1.0 + np.exp(-rate * (s - s0)))


def logit(
    s: NDArrayFloat, s0: float = 0.0, rate: float = 1.0, supremum: float = 1.0
) -> NDArrayFloat:
    """
    Return the logit function (inverse to logistic).

    Parameters
    ----------
    s : NDArrayFloat
        Input values.
    s0 : float, optional
        Value of the function's midpoint, by default 0.0?
    rate : float, optional
        The logistic growth rate or steepness of the curve, by default 1.0
    supremum : float, optional
        The supremum of the values of the function, by default 1.0

    Returns
    -------
    NDArrayFloat
        Logistic values.
    """
    return -np.log(supremum / s - 1.0) / rate + s0


def tanh_wrapper(
    s: NDArrayFloat, s0: float = 0.0, rate: float = 1.0, supremum: float = 1.0
) -> NDArrayFloat:
    """
    Return the hyperbolic tangent function (inverse to arctanh).

    Parameters
    ----------
    s : NDArrayFloat
        Input values.
    s0 : float, optional
        Value of the function's midpoint, by default 0.0?
    rate : float, optional
        The logistic growth rate or steepness of the curve, by default 1.0
    supremum : float, optional
        The supremum of the values of the function, by default 1.0

    Returns
    -------
    NDArrayFloat
        Logistic values.
    """
    return supremum / 2.0 * (np.tanh((s - s0) * rate) + 1.0)


def dtanh_wrapper(
    s: NDArrayFloat, s0: float = 0.0, rate: float = 1.0, supremum: float = 1.0
) -> NDArrayFloat:
    """
    Return the derivative (w.r.t. s) of the hyperbolic tangent function.

    Parameters
    ----------
    s : NDArrayFloat
        Input values.
    s0 : float, optional
        Value of the function's midpoint, by default 0.0?
    rate : float, optional
        The logistic growth rate or steepness of the curve, by default 1.0
    supremum : float, optional
        The supremum of the values of the function, by default 1.0

    Returns
    -------
    NDArrayFloat
        Logistic values.
    """
    return supremum / 2.0 * rate * (1 - np.tanh((s - s0) * rate) ** 2)


def arctanh_wrapper(
    s: NDArrayFloat, s0: float = 0.0, rate: float = 1.0, supremum: float = 1.0
) -> NDArrayFloat:
    """
    Return the inverse of the hyperbolic tangent function.

    Parameters
    ----------
    s : NDArrayFloat
        Input values.
    s0 : float, optional
        Value of the function's midpoint, by default 0.0?
    rate : float, optional
        The logistic growth rate or steepness of the curve, by default 1.0
    supremum : float, optional
        The supremum of the values of the function, by default 1.0

    Returns
    -------
    NDArrayFloat
        Logistic values.
    """
    with np.errstate(divide="ignore"):
        return np.arctanh(s / supremum * 2.0 - 1.0) / rate + s0


def darctanh_wrapper(
    s: NDArrayFloat, s0: float = 0.0, rate: float = 1.0, supremum: float = 1.0
) -> NDArrayFloat:
    """
    Return the derivative (w.r.t. s) of the inverse of the hyperbolic tangent function.

    Parameters
    ----------
    s : NDArrayFloat
        Input values.
    s0 : float, optional
        Value of the function's midpoint, by default 0.0?
    rate : float, optional
        The logistic growth rate or steepness of the curve, by default 1.0
    supremum : float, optional
        The supremum of the values of the function, by default 1.0

    Returns
    -------
    NDArrayFloat
        Logistic values.
    """
    with np.errstate(divide="ignore"):
        return -supremum / (2.0 * rate * s * (s - supremum))


def to_new_range(
    s_raw: NDArrayFloat,
    old_lbound: float,
    old_ubound: float,
    new_lbound: float,
    new_ubound: float,
    is_log10: bool = False,
) -> NDArrayFloat:
    """
    Rescale the input values to the new desired range.

    Parameters
    ----------
    s_raw : NDArrayFloat
        Input values to rescale.
    old_lbound : float
        Input range lower bound.
    old_ubound : float
        Input range upper bound.
    new_lbound : float
        New range lower bound.
    new_ubound : float
        New range upper bound.
    is_log: bool
        Whether to use a log10 scale for the new range.

    Returns
    -------
    NDArrayFloat
        Rescaled output values.
    """
    if is_log10:
        return 10.0 ** to_new_range(
            s_raw,
            old_lbound,
            old_ubound,
            np.log10(new_lbound),
            np.log10(new_ubound),
            is_log10=False,
        )
    return (s_raw - old_lbound) * (new_ubound - new_lbound) / (
        float(old_ubound) - float(old_lbound)
    ) + new_lbound


def to_new_range_derivative(
    s_raw: NDArrayFloat,
    old_lbound: float,
    old_ubound: float,
    new_lbound: float,
    new_ubound: float,
    is_log10: bool = False,
) -> NDArrayFloat:
    """
    Rescale the input values to the new desired range.

    Parameters
    ----------
    s_raw : NDArrayFloat
        Input values to rescale.
    old_lbound : float
        Input range lower bound.
    old_ubound : float
        Input range upper bound.
    new_lbound : float
        New range lower bound.
    new_ubound : float
        New range upper bound.
    is_log: bool
        Whether to use a log10 scale for the new range.

    Returns
    -------
    NDArrayFloat
        Rescaled output values.
    """
    # we use recursivity
    if not is_log10:
        return np.array(
            [(new_ubound - new_lbound) / (float(old_ubound) - float(old_lbound))]
        )

    deriv = to_new_range(
        s_raw,
        old_lbound,
        old_ubound,
        np.log10(new_lbound),
        np.log10(new_ubound),
        is_log10=False,
    )
    return (
        np.log(10)
        * 10 ** (deriv)
        * (np.log10(new_ubound) - np.log10(new_lbound))
        / (float(old_ubound) - float(old_lbound))
    )


class RangeRescaler(Preconditioner):
    """
    Rescale the values from the old range to the new desired range.

    The log10 option allows to work with non linearly scaled parameters such as
    diffusivity or permeability.
    """

    def __init__(
        self,
        old_lbound: float,
        old_ubound: float,
        new_lbound: float,
        new_ubound: float,
        is_log10: bool = False,
    ) -> None:
        """
        Initialize the instance.

        Parameters
        ----------
        lbound : NDArrayFloat
            Lower bound of the original values.
        ubound : NDArrayFloat
            Upper bound of the original values.
        lbound : NDArrayFloat
            Lower bound of the conditioned values. This does not really matters.
            The default is -5.0.
        ubound : NDArrayFloat
            Upper bound of the conditioned values. This does not really matters.
            The default is 5.0.
        is_log: bool
            Whether to use a log10-scaling for the logit y-scale.
        """
        self.old_lbound: float = old_lbound
        self.old_ubound: float = old_ubound
        self.new_lbound = new_lbound
        self.new_ubound = new_ubound
        self.is_log10: bool = is_log10

    def _transform(self, s_raw: NDArrayFloat) -> NDArrayFloat:
        """
        Apply the preconditioning/parametrization.

        Parameters
        ----------
        s_raw : NDArrayFloat
            Non-conditioned (transformed) parameter values.

        Returns
        -------
        NDArrayFloat
            Conditioned (transformed) parameter values.
        """
        # ici il faut faire l'opposé
        # rescaling between 5 and -5 -> the scale does not really matter
        return to_new_range(
            np.log(s_raw) / np.log(10.0) if self.is_log10 else s_raw,
            old_lbound=(
                np.log(self.old_lbound) / np.log(10.0)
                if self.is_log10
                else self.old_lbound
            ),
            old_ubound=(
                np.log(self.old_ubound) / np.log(10.0)
                if self.is_log10
                else self.old_ubound
            ),
            new_lbound=self.new_lbound,
            new_ubound=self.new_ubound,
        )

    def _backtransform(self, s_cond: NDArrayFloat) -> NDArrayFloat:
        """
        Apply the back-preconditioning/parametrization.

        Parameters
        ----------
        s_cond : NDArrayFloat
            Conditioned (transformed) parameter values.

        Returns
        -------
        NDArrayFloat
            Non-conditioned (transformed) parameter values.
        """
        return to_new_range(
            s_cond,
            old_lbound=self.new_lbound,
            old_ubound=self.new_ubound,
            new_lbound=self.old_lbound,
            new_ubound=self.old_ubound,
            is_log10=self.is_log10,
        )

    def _dtransform_vec(
        self, s_raw: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat:
        """
        Return the transform 1st derivative times a vector as a 1-D vector..
        """
        # we apply the chain rule
        return (
            to_new_range_derivative(
                np.log(s_raw) / np.log(10.0) if self.is_log10 else s_raw,
                old_lbound=(
                    np.log(self.old_lbound) / np.log(10.0)
                    if self.is_log10
                    else self.old_lbound
                ),
                old_ubound=(
                    np.log(self.old_ubound) / np.log(10.0)
                    if self.is_log10
                    else self.old_ubound
                ),
                new_lbound=self.new_lbound,
                new_ubound=self.new_ubound,
            )
            / (s_raw * np.log(10.0) if self.is_log10 else 1.0)
            * gradient
        )

    def _dbacktransform_vec(
        self, s_cond: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat:
        """
        Return the backtransform 1st derivative times a vector.
        """
        return (
            to_new_range_derivative(
                s_cond,
                old_lbound=self.new_lbound,
                old_ubound=self.new_ubound,
                new_lbound=self.old_lbound,
                new_ubound=self.old_ubound,
                is_log10=self.is_log10,
            )
        ) * gradient

    def _dbacktransform_inv_vec(
        self, s_cond: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat:
        """
        Return the inverse of the backtransform 1st derivative times a vector.
        """
        return gradient / (
            to_new_range_derivative(
                s_cond,
                old_lbound=self.new_lbound,
                old_ubound=self.new_ubound,
                new_lbound=self.old_lbound,
                new_ubound=self.old_ubound,
                is_log10=self.is_log10,
            )
        )


class SigmoidRescaler(Preconditioner):
    """
    Rescale the values using a sigmoid transform.

    The underlying function is an hyperbolic tangent (tanh). Raw values must be
    in the interval ]0, 1[.

    This parametrization is particularly useful when a parameter
    has two modes that are the limits of the value page. For example,
    to image a porosity field and two facies, one porous, the other not
    with more or less homogeneous values for each facies (e.g. 15% and 40% ).

    The log10 option allows to work with non linearly scaled parameters such as
    diffusivity or permeability.
    """

    LBOUND_RAW = 0.0
    UBOUND_RAW = 1.0

    def __init__(
        self, s0: float = 0.0, rate: float = 1.0, supremum: float = 1.0
    ) -> None:
        """
        Initialize the instance.

        Parameters
        ----------
        rate: float
            Growth rate. The higher the rate, the steeper the sigmoid. A value of
            3 is usually the upper acceptable limit, i.e., above this value,
            the bijection between "transform" and "backtransform" might be lost and the
            derivative might become ncorrect.
        """
        self.s0: float = s0
        self.rate: float = rate
        self.supremum: float = supremum

    def _transform(self, s_raw: NDArrayFloat) -> NDArrayFloat:
        """
        Apply the preconditioning/parametrization.

        Parameters
        ----------
        s_raw : NDArrayFloat
            Non-conditioned (transformed) parameter values.

        Returns
        -------
        NDArrayFloat
            Conditioned (transformed) parameter values.
        """
        return arctanh_wrapper(
            s_raw, s0=self.s0, rate=self.rate, supremum=self.supremum
        )

    def _backtransform(self, s_cond: NDArrayFloat) -> NDArrayFloat:
        """
        Apply the back-preconditioning/parametrization.

        Parameters
        ----------
        s_cond : NDArrayFloat
            Conditioned (transformed) parameter values.

        Returns
        -------
        NDArrayFloat
            Non-conditioned (transformed) parameter values.
        """
        return tanh_wrapper(s_cond, s0=self.s0, rate=self.rate, supremum=self.supremum)

    def _dtransform_vec(
        self, s_raw: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat:
        """
        Return the transform 1st derivative times a vector as a 1-D vector..
        """
        return (
            darctanh_wrapper(s_raw, s0=self.s0, rate=self.rate, supremum=self.supremum)
            * gradient
        )

    def _dbacktransform_vec(
        self, s_cond: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat:
        """
        Return the backtransform 1st derivative times a vector.
        """
        # otherwise chain rule of two functions
        return (
            dtanh_wrapper(s_cond, s0=self.s0, rate=self.rate, supremum=self.supremum)
            * gradient
        )

    def _dbacktransform_inv_vec(
        self, s_cond: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat:
        """
        Return the inverse of the backtransform 1st derivative times a vector.
        """
        return gradient / (
            dtanh_wrapper(s_cond, s0=self.s0, rate=self.rate, supremum=self.supremum)
        )


class SigmoidRescalerBounded(ChainedTransforms):
    LBOUND_RAW = -np.inf  # cannot be zero
    UBOUND_RAW = +np.inf
    LBOUND_COND: float = -10
    UBOUND_COND: float = +10

    def __init__(
        self,
        old_lbound: float,
        old_ubound: float,
        rate: float,
        is_log10: bool,
    ) -> None:
        """
        Initialize the instance.

        Parameters
        ----------
        old_lbound : NDArrayFloat
            Lower bound of the original values.
        old_ubound : NDArrayFloat
            Upper bound of the original values.
        rate: float
            Growth rate. The higher the rate, the steeper the sigmoid. A value of
            3 is usually the upper acceptable limit, i.e., above this value,
            the bijection between "transform" and "backtransform" might be lost and the
            derivative might become ncorrect.
        is_log: bool
            Whether to use a log10-scaling for the logit y-scale.
        lbound : NDArrayFloat
            Lower bound of the conditioned values. This does not really matters.
            The default is -5.0.
        ubound : NDArrayFloat
            Upper bound of the conditioned values. This does not really matters.
            The default is 5.0.
        """
        self.pcds = [
            RangeRescaler(old_lbound, old_ubound, 0.0, 1.0, is_log10),
            SigmoidRescaler(0.0, rate, 1.0),
        ]

    def transform_bounds(self, bounds: NDArrayFloat) -> NDArrayFloat:
        """
        Transform the bounds to match the preconditioned values.

        Parameters
        ----------
        bounds : NDArrayFloat
            Array of shape (N_s, 2).

        Returns
        -------
        NDArrayFloat
            Array of shape (N_s, 2) with transformed bounds.
        """
        return np.array(
            [
                np.ones_like(bounds[:, 0]) * self.LBOUND_COND,
                np.ones_like(bounds[:, 0]) * self.UBOUND_COND,
            ]
        ).T


class Normalizer(Preconditioner):
    """Apply a sqrt preconditioning to ensure positive values of the parameter."""

    def __init__(self, s_prior: NDArrayFloat) -> None:
        """

        Parameters
        ----------
        s_prior : NDArrayFloat
            _description_
        """
        super().__init__()
        # mean value of the prior field
        self.prior_mean = np.mean(s_prior)
        # standard deviation of the prior field
        self.prior_std = np.std(s_prior)

    def _transform(self, s_raw: NDArrayFloat) -> NDArrayFloat:
        """
        Apply the preconditioning/parametrization.

        Parameters
        ----------
        s_raw : NDArrayFloat
            Non-conditioned (transformed) parameter values.

        Returns
        -------
        NDArrayFloat
            Conditioned (transformed) parameter values.
        """
        return (s_raw - self.prior_mean) / self.prior_std

    def _backtransform(self, s_cond: NDArrayFloat) -> NDArrayFloat:
        """
        Apply the back-preconditioning/parametrization.

        Parameters
        ----------
        s_cond : NDArrayFloat
            Conditioned (transformed) parameter values.

        Returns
        -------
        NDArrayFloat
            Non-conditioned (transformed) parameter values.
        """
        return s_cond * self.prior_std + self.prior_mean

    def _dtransform_vec(
        self, s_raw: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat:
        """
        Return the transform 1st derivative times a vector as a 1-D vector..
        """
        return gradient / self.prior_std

    def _dbacktransform_vec(
        self, s_cond: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat:
        """
        Return the backtransform 1st derivative times a vector.
        """
        return self.prior_std * gradient

    def _dbacktransform_inv_vec(
        self, s_cond: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat:
        """
        Return the inverse of the backtransform 1st derivative times a vector.
        """
        return gradient / self.prior_std


class StdRescaler(Preconditioner):
    """Apply a sqrt preconditioning to ensure positive values of the parameter."""

    def __init__(
        self,
        s_prior: NDArrayFloat,
        prior_std: Optional[float] = None,
    ) -> None:
        super().__init__()
        # need to store the prior field for the rescaling
        self.s_prior = s_prior

        # store to avoid computing many times
        if prior_std is None:
            self.prior_std: float = float(np.std(self.s_prior))
        else:
            self.prior_std = prior_std

    def _transform(self, s_raw: NDArrayFloat) -> NDArrayFloat:
        """
        Apply the preconditioning/parametrization.

        Parameters
        ----------
        s_raw : NDArrayFloat
            Non-conditioned (transformed) parameter values.

        Returns
        -------
        NDArrayFloat
            Conditioned (transformed) parameter values.
        """
        return (s_raw - self.s_prior) / self.prior_std

    def _backtransform(self, s_cond: NDArrayFloat) -> NDArrayFloat:
        """
        Apply the back-preconditioning/parametrization.

        Parameters
        ----------
        s_cond : NDArrayFloat
            Conditioned (transformed) parameter values.

        Returns
        -------
        NDArrayFloat
            Non-conditioned (transformed) parameter values.
        """
        return s_cond * self.prior_std + self.s_prior

    def _dtransform_vec(
        self, s_raw: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat:
        """
        Return the transform 1st derivative times a vector as a 1-D vector..
        """
        return gradient / self.prior_std

    def _dbacktransform_vec(
        self, s_cond: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat:
        """
        Return the backtransform 1st derivative times a vector.
        """
        return self.prior_std * gradient

    def _dbacktransform_inv_vec(
        self, s_cond: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat:
        """
        Return the inverse of the backtransform 1st derivative times a vector.
        """
        return gradient / self.prior_std


class BoundsRescaler(Preconditioner):
    """Apply a bound rescaling (aka logit)."""

    EPSILON = 1e-10

    def __init__(
        self,
        lbounds: Union[float, NDArrayFloat],
        ubounds: Union[float, NDArrayFloat],
    ) -> None:
        super().__init__()
        # need to store bounds for the rescaling process
        self.lbounds = lbounds
        self.ubounds = ubounds

    def _transform(self, s_raw: NDArrayFloat) -> NDArrayFloat:
        """
        Apply the preconditioning/parametrization.

        Parameters
        ----------
        s_raw : NDArrayFloat
            Non-conditioned (transformed) parameter values.

        Returns
        -------
        NDArrayFloat
            Conditioned (transformed) parameter values.
        """
        # clip to bounds with very small values to avoid negative values in the log
        _s_raw = s_raw.clip(self.lbounds + self.EPSILON, self.ubounds - self.EPSILON)
        return np.log((_s_raw - self.lbounds) / (self.ubounds - _s_raw))

    def _backtransform(self, s_cond: NDArrayFloat) -> NDArrayFloat:
        """
        Apply the back-preconditioning/parametrization.

        Parameters
        ----------
        s_cond : NDArrayFloat
            Conditioned (transformed) parameter values.

        Returns
        -------
        NDArrayFloat
            Non-conditioned (transformed) parameter values.
        """
        bounds_mean = 0.5 * (self.ubounds + self.lbounds)
        bounds_half_amplitude = 0.5 * (self.ubounds - self.lbounds)
        return bounds_mean + bounds_half_amplitude * (
            (np.exp(s_cond) - 1) / (np.exp(s_cond) + 1)
        )

    def _dtransform_vec(
        self, s_raw: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat:
        """
        Return the transform 1st derivative times a vector as a 1-D vector..
        """
        _s_raw = s_raw.clip(self.lbounds + self.EPSILON, self.ubounds - self.EPSILON)
        return (
            -(self.ubounds - self.lbounds)
            / ((_s_raw - self.lbounds) * (_s_raw - self.ubounds))
            * gradient
        )

    def _dbacktransform_vec(
        self, s_cond: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat:
        """
        Return the backtransform 1st derivative times a vector.
        """
        return (
            (self.ubounds - self.lbounds) * np.exp(s_cond) / (np.exp(s_cond) + 1) ** 2
        ) * gradient

    def _dbacktransform_inv_vec(
        self, s_cond: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat:
        """
        Return the inverse of the backtransform 1st derivative times a vector.
        """
        return gradient / (
            (self.ubounds - self.lbounds) * np.exp(s_cond) / (np.exp(s_cond) + 1) ** 2
        )


def get_gd_weights(theta: NDArrayFloat) -> NDArrayFloat:
    if np.size(theta) < 1:
        raise ValueError("The theta vector is empty!")
    # initialize the vector of weights
    weights: NDArrayFloat = np.zeros((theta.size + 1))
    # first weight
    weights[0] = np.prod(np.cos(theta))
    # i = 1... Ne - 2
    for i in range(1, theta.size):
        weights[i] = np.sin(theta[i - 1]) * np.prod(np.cos(theta[i:]))
    # last weight
    weights[-1] = np.sin(theta[-1])
    # ensure that the sum of squared weights equals to one
    np.testing.assert_almost_equal(np.sum(weights**2), 1.0)
    return weights


def gd_parametrize(W: NDArrayFloat, weights: NDArrayFloat) -> NDArrayFloat:
    """
    Apply the gradual deformation parametrization to generate a new Z.

    Z is a random variable following a centered-reduced normal distribution.

    Parameters
    ----------
    z_arr : NDArrayFloat
        Array with size (N_z, Ne) which columns are independent random variables
        following a centered-reduced normal distribution of size (N_z).
        Ne is the number of independent realizations.
    weights : NDArrayFloat
        Gradual deformation weights.

    Returns
    -------
    NDArrayFloat
        New random variable following a centered-reduced normal distribution.
    """
    return np.sum(W * weights, axis=1)


def d_gd_parametrize_mat_vec(
    z_arr: NDArrayFloat, theta: NDArrayFloat, b: NDArrayFloat
) -> NDArrayFloat:
    """
    Return the GD parametrization derivative w.r.t. theta times a vector.

    Parameters
    ----------
    z_arr : NDArrayFloat
        Array with size (N_z, Ne) which columns are independent random variables
        following a centered-reduced normal distribution of size (N_z).
        Ne is the number of independent realizations.
    theta : NDArrayFloat
        Gradual deformation parameter.

    Returns
    -------
    NDArrayFloat
        New random variable following a centered-reduced normal distribution.
    """
    # check input size
    assert np.size(b) == np.size(z_arr[:, 0])

    dweights: NDArrayFloat = -np.prod(np.cos(theta)) / np.cos(theta) * np.sin(theta)
    # z0
    dz = dweights * (z_arr[:, 0] @ b)
    # z_ne -> the derivative only apply to the last element of \theta
    dweight = np.cos(theta[-1])
    dz[-1] += dweight * z_arr[:, -1] @ b

    # i = 1... Ne - 2
    for i in range(1, z_arr.shape[-1] - 1):
        v = z_arr[:, i] @ b
        # product
        tmp_prod = np.prod(np.cos(theta[i:]))
        # dealing with the sin
        dz[i - 1] += np.cos(theta[i - 1]) * tmp_prod * v
        # dealing with the cos product
        dz[i:] -= (
            np.sin(theta[i - 1]) * tmp_prod / np.cos(theta[i:]) * np.sin(theta[i:]) * v
        )

    assert np.size(dz) == np.size(theta)
    return dz


def _check_ne(ne: int) -> int:
    try:
        ne = int(ne)
        if ne < 2:
            raise ValueError
        return ne
    except (TypeError, ValueError) as err:
        raise ValueError("ne must be an integer, >=2.") from err


def get_theta_init_normal(
    ne: int,
    mu: float = 0.5,
    sigma: float = 0.15,
    random_state: Optional[
        Union[int, np.random.Generator, np.random.RandomState]
    ] = None,
) -> NDArrayFloat:
    """
    Get the initial theta vector such that ai are drawn from a normal distribution.

    Parameters
    ----------
    ne : int
        Number of realizations
    mu: float
        Mean of the normal distribution for a_i. The default is 0.5.
    sigma: float
        Standard deviation of the normal distribution for a_i. The default is 0.15.
    random_state : Optional[Union[int, np.random.Generator, np.random.RandomState]]
        Pseudorandom number generator state used to generate resamples.
        If `random_state` is ``None`` (or `np.random`), the
        `numpy.random.RandomState` singleton is used.
        If `random_state` is an int, a new ``RandomState`` instance is used,
        seeded with `random_state`.
        If `random_state` is already a ``Generator`` or ``RandomState``
        instance then that instance is used. The default is None

    Returns
    -------
    NDArrayFloat
        Vector of param theta with size (ne - 1).
    """
    a: NDArrayFloat = check_random_state(random_state).normal(
        loc=mu, scale=sigma, size=_check_ne(ne)
    )
    return get_theta_init(a / np.linalg.norm(a))


def get_theta_init_uniform(ne: int) -> NDArrayFloat:
    """
    Get the initial theta vector so all weights ai are the same (1/ne).

    Parameters
    ----------
    ne : int
        Number of realizations

    Returns
    -------
    NDArrayFloat
        Vector of param theta with size (ne - 1).
    """
    ne = _check_ne(ne)
    return get_theta_init(np.ones(ne) / np.sqrt(ne))


def get_theta_init(target_weights: NDArrayFloat) -> NDArrayFloat:
    """
    Get the initial theta vector to ensure the given target weights.

    Parameters
    ----------
    target_weights : NDArrayFloat
        Target weights with size (Ne,).

    Returns
    -------
    NDArrayFloat
        Vector of param theta with size (ne - 1).
    """
    ne = np.size(target_weights)
    params = np.zeros((ne - 1))
    params[-1] = np.arcsin(target_weights[-1])

    # i = 1... Ne - 2
    for i in range(ne - 3, -1, -1):
        params[i] = np.arcsin(target_weights[i + 1] / np.prod(np.cos(params[i + 1 :])))
    return params


class GDPNCS(Preconditioner):
    """
    Apply a Gradual Deformation parametrization associated with the SPDE approach.

    The Gradual Deformation parametrization is used to generate a white noise
    reduced and centered as a linear combinaison of Ne white noises while adjusting
    Ne-1 parameters. The obtained white noise is used in the SPDE approach to
    generate a field with the required geostatistical parameters.

    Here: non conditional-simulation.

    TODO: add ref.
    """

    def __init__(
        self,
        ne: int,
        Q_nc: csc_array,
        estimated_mean: float = 0.0,
        theta: Optional[NDArrayFloat] = None,
        cholQ_nc: Optional[Factor] = None,
        random_state: Optional[
            Union[int, np.random.Generator, np.random.RandomState]
        ] = None,
        is_update_mean: bool = True,
    ) -> None:
        """
        Initialize the instance.

        Parameters
        ----------
        random_state : Optional[Union[int, np.random.Generator, np.random.RandomState]]
            Pseudorandom number generator state used to generate resamples.
            If `random_state` is ``None`` (or `np.random`), the
            `numpy.random.RandomState` singleton is used.
            If `random_state` is an int, a new ``RandomState`` instance is used,
            seeded with `random_state`.
            If `random_state` is already a ``Generator`` or ``RandomState``
            instance then that instance is used. The default is None
        """
        # initialize the super instance
        super().__init__()

        self.estimated_mean: float = estimated_mean
        self.is_update_mean: bool = is_update_mean

        # TODO: check if Q_nc is square

        # perform the cholesky factorization of the sparse non conditional precision
        # matrix for fast inversion
        if cholQ_nc is None:
            self._cholQ_nc = sparse_cholesky(Q_nc)
        else:
            self._cholQ_nc = cholQ_nc

        # initialize the ensemble of white noises with shape (Ns, Ne)
        self.W: NDArrayFloat = check_random_state(random_state).normal(
            size=(Q_nc.shape[0], ne)
        )

        if theta is not None:
            # check the length correctness
            assert np.size(theta) == ne - 1
            _theta: NDArrayFloat = theta
        else:
            # initialize theta such as weights (ai) are all equals (1/sqrt(Ne)).
            _theta = get_theta_init_uniform(ne)
        self.theta = _theta

    def _get_test_data(
        self,
        lbounds: Union[float, NDArrayFloat],
        ubounds: Union[float, NDArrayFloat],
        shape: Optional[Union[int, Sequence[int]]] = None,
    ) -> NDArrayFloat:
        """
        Get test data to check the preconditioner correctness.

        This is a development tool.
        """
        if self.is_update_mean:
            return self.backtransform(np.hstack((self.theta, self.estimated_mean)))
        return self.backtransform(self.theta)

    def _transform(self, s_raw: NDArrayFloat) -> NDArrayFloat:
        """
        Apply the preconditioning/parametrization.

        It works a bit differently for this preconditioner -> the value is not taken
        from the model.

        Parameters
        ----------
        s_raw : NDArrayFloat
            Non-conditioned (transformed) parameter values.

        Returns
        -------
        NDArrayFloat
            Conditioned (transformed) parameter values.
        """
        assert s_raw.size == self._cholQ_nc.P().size
        if self.is_update_mean:
            return np.hstack((self.theta, self.estimated_mean))
        return self.theta

    def _backtransform(self, s_cond: NDArrayFloat) -> NDArrayFloat:
        """
        Apply the back-preconditioning/parametrization.

        Parameters
        ----------
        s_cond : NDArrayFloat
            Conditioned (transformed) parameter values.

        Returns
        -------
        NDArrayFloat
            Non-conditioned (transformed) parameter values.
        """
        if self.is_update_mean:
            self.theta = s_cond[:-1]
            self.estimated_mean = s_cond[-1]
        else:
            self.theta = s_cond

        return (
            spde.simu_nc(
                self._cholQ_nc,
                w=gd_parametrize(self.W, get_gd_weights(self.theta)),
            )
            + self.estimated_mean
        )

    def _dtransform_vec(
        self, s_raw: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat:
        """
        Return the transform 1st derivative times a vector as a 1-D vector..
        """
        return np.zeros_like(s_raw)

    def _dbacktransform_vec(
        self, s_cond: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat:
        """
        Return the backtransform 1st derivative times a vector.
        """
        out = d_gd_parametrize_mat_vec(
            self.W, self.theta, spde.d_simu_nc_mat_vec(self._cholQ_nc, gradient)
        )
        if self.is_update_mean:
            return np.hstack((out, np.sum(gradient)))
        return out

    def _dbacktransform_inv_vec(
        self, s_cond: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat:
        """
        Return the inverse of the backtransform 1st derivative times a vector.
        """
        raise NotImplementedError(
            "_dbacktransform_inv_vec is not implemented for "
            "GDPNCS! Contact developers for detail!"
        )

    def transform_bounds(self, bounds: NDArrayFloat) -> NDArrayFloat:
        """
        Transform the bounds to match the preconditioned values.

        Parameters
        ----------
        bounds : NDArrayFloat
            Array of shape (N_s, 2).

        Returns
        -------
        NDArrayFloat
            Array of shape (N_s, 2) with transformed bounds.
        """
        n_cond = self.theta.size
        if self.is_update_mean:
            n_cond += 1
        bounds = np.zeros((n_cond, 2))
        bounds[:, 0] = -np.inf
        bounds[:, 1] = np.inf
        return bounds


class GDPCS(GDPNCS):
    """
    Apply a Gradual Deformation parametrization associated with the SPDE approach.

    The Gradual Deformation parametrization is used to generate a white noise
    reduced and centered as a linear combinaison of Ne white noises while adjusting
    Ne-1 parameters. The obtained white noise is used in the SPDE approach to
    generate a field with the required geostatistical parameters.

    Here: conditional-simulation.

    TODO: add ref.
    """

    def __init__(
        self,
        ne: int,
        Q_nc: csc_array,
        Q_c: csc_array,
        estimated_mean: float,
        dat_nn: NDArrayInt,
        dat_val: NDArrayFloat,
        dat_var: NDArrayFloat,
        theta: Optional[NDArrayFloat] = None,
        cholQ_nc: Optional[Factor] = None,
        cholQ_c: Optional[Factor] = None,
        random_state: Optional[
            Union[int, np.random.Generator, np.random.RandomState]
        ] = None,
        is_update_mean: bool = True,
    ) -> None:
        """
        Initialize the instance.

        Parameters
        ----------
        random_state : Optional[Union[int, np.random.Generator, np.random.RandomState]]
            Pseudorandom number generator state used to generate resamples.
            If `random_state` is ``None`` (or `np.random`), the
            `numpy.random.RandomState` singleton is used.
            If `random_state` is an int, a new ``RandomState`` instance is used,
            seeded with `random_state`.
            If `random_state` is already a ``Generator`` or ``RandomState``
            instance then that instance is used. The default is None
        """
        # initialize the super instance
        super().__init__(
            ne=ne,
            Q_nc=Q_nc,
            estimated_mean=estimated_mean,
            theta=theta,
            cholQ_nc=cholQ_nc,
            random_state=random_state,
            is_update_mean=is_update_mean,
        )

        # perform the cholesky factorization of the sparse non conditional precision
        # matrix for fast inversion
        if cholQ_c is None:
            self._cholQ_c = sparse_cholesky(Q_c)
        else:
            self._cholQ_c = cholQ_c
        self._Q_c = Q_c

        # Conditioning data
        self.dat_nn = dat_nn
        self.dat_val = dat_val
        self.dat_var = dat_var

    def _backtransform(self, s_cond: NDArrayFloat) -> NDArrayFloat:
        """
        Apply the back-preconditioning/parametrization.

        Parameters
        ----------
        s_cond : NDArrayFloat
            Conditioned (transformed) parameter values.

        Returns
        -------
        NDArrayFloat
            Non-conditioned (transformed) parameter values.
        """
        if self.is_update_mean:
            self.theta = s_cond[:-1]
            self.estimated_mean = s_cond[-1]
        else:
            self.theta = s_cond
        return (
            spde.simu_c(
                self._cholQ_nc,
                self._Q_c,
                self._cholQ_c,
                self.dat_val - self.estimated_mean,
                self.dat_nn,
                self.dat_var,
                w=gd_parametrize(self.W, get_gd_weights(self.theta)),
            )
            + self.estimated_mean
        )

    def _dtransform_vec(
        self, s_raw: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat:
        """
        Return the transform 1st derivative times a vector as a 1-D vector..
        """
        return np.zeros_like(s_raw)

    def _dbacktransform_vec(
        self, s_cond: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat:
        """
        Return the transform 1st derivative times a vector as a 1-D vector..
        """
        out = d_gd_parametrize_mat_vec(
            self.W,
            self.theta,
            spde.d_simu_c_matvec(
                self._cholQ_nc, self._cholQ_c, self.dat_nn, self.dat_var, gradient
            ),
        )
        if self.is_update_mean:
            Z = lil_array((self._cholQ_nc.L().shape[0], self.dat_nn.size))
            Z[self.dat_nn, np.arange(self.dat_nn.size)] = 1

            return np.hstack(
                (
                    out,
                    np.sum(gradient)
                    - (1 / self.dat_var @ (Z.T @ self._cholQ_c(gradient))),
                )
            )
        return out

    def _dbacktransform_inv_vec(
        self, s_cond: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat:
        """
        Return the inverse of the backtransform 1st derivative times a vector.
        """
        raise NotImplementedError(
            "_dbacktransform_inv_vec is not implemented for "
            "GDPCS! Contact developers for detail!"
        )


class SubSelector(Preconditioner):
    """Apply a selection on the input field."""

    def __init__(self, node_numbers: NDArrayInt, geometry: Geometry) -> None:
        """
        Initialize the instance.

        Parameters
        ----------
        node_numbers : NDArrayInt
            Node to sample.
        field_shape : int
            Size of the field to be sampled.
        """
        self.node_numbers = np.array(node_numbers)
        self.field_size: int = geometry.n_grid_cells
        self.s_raw = np.zeros(self.field_size)

    def _transform(self, s_raw: NDArrayFloat) -> NDArrayFloat:
        """
        Apply the preconditioning/parametrization.

        Parameters
        ----------
        s_raw : NDArrayFloat
            Non-conditioned (transformed) parameter values.

        Returns
        -------
        NDArrayFloat
            Conditioned (transformed) parameter values.
        """
        self.s_raw = s_raw
        assert np.size(s_raw) == self.field_size
        return self.s_raw[self.node_numbers]  # ! this is 1D.

    def _backtransform(self, s_cond: NDArrayFloat) -> NDArrayFloat:
        """
        Apply the back-preconditioning/parametrization.

        Parameters
        ----------
        s_cond : NDArrayFloat
            Conditioned (transformed) parameter values.

        Returns
        -------
        NDArrayFloat
            Non-Conditioned (transformed) parameter values.
        """
        assert np.size(s_cond) == self.node_numbers.size
        out = self.s_raw.copy()
        out[self.node_numbers] = s_cond
        return out

    def _dtransform_vec(
        self, s_raw: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat:
        """
        Return the transform 1st derivative times a vector as a 1-D vector..
        """
        assert np.size(s_raw) == self.field_size
        assert np.size(gradient) == self.node_numbers.size
        out = np.zeros(self.field_size)
        out[self.node_numbers] = gradient
        return out

    def _dbacktransform_vec(
        self, s_cond: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat:
        """
        Return the backtransform 1st derivative times a vector.
        """
        assert gradient.size == self.field_size
        return gradient[self.node_numbers]

    def _dbacktransform_inv_vec(
        self, s_cond: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat:
        """
        Return the inverse of the backtransform 1st derivative times a vector.
        """
        # no effect here. We just sub sample the gradient
        assert gradient.size == self.node_numbers.size
        assert s_cond.size == self.node_numbers.size
        out = np.zeros(self.field_size)
        out[self.node_numbers] = gradient
        return out

    def transform_bounds(self, bounds: NDArrayFloat) -> NDArrayFloat:
        """
        Transform the bounds to match the preconditioned values.

        Parameters
        ----------
        bounds : NDArrayFloat
            Array of shape (N_s, 2).

        Returns
        -------
        NDArrayFloat
            Array of shape (N_s, 2) with transformed bounds.
        """
        # store s_raw
        s_raw = self.s_raw
        # calling transform_bounds modify s_raw (set at the upper bound)
        # which is not necessarily desired
        bounds = super().transform_bounds(bounds)
        # restore s_raw
        self.s_raw = s_raw
        # return the bounds
        return bounds

    def test_preconditioner(
        self,
        lbounds: Union[float, NDArrayFloat],
        ubounds: Union[float, NDArrayFloat],
        shape: Optional[Union[int, Sequence[int]]] = None,
        rtol: float = 1e-5,
        eps: Optional[float] = None,
    ) -> None:
        """
        Test if the backconditioner and the derivatives times a vector are correct.

        This is a development tool.

        Parameters
        ----------
        lbounds : Union[float, NDArrayFloat]
            _description_
        ubounds : Union[float, NDArrayFloat]
            _description_
        shape : Optional[Union[int, Sequence[int]]], optional
            _description_, by default None
        rtol : float, optional
            _description_, by default 1e-5
        eps : Optional[float], optional
            The epsilon for the computation of the approximated preconditioner first
            derivative by finite difference. by default None.

        Raises
        ------
        ValueError
            If one of the backconditioner of the gradient conditioner are incorrect.
        """
        super()._test_preconditioner(lbounds, ubounds, shape, rtol, eps, [4])


class Slicer(SubSelector):
    """Apply a slicing to the field of values."""

    def __init__(
        self,
        geometry: Geometry,
        span: Union[NDArrayInt, Tuple[slice, slice], NDArrayBool] = (
            slice(None),
            slice(None),
        ),
    ) -> None:
        """Initialize the instance."""
        field_size = geometry.n_grid_cells
        node_numbers = np.arange(field_size).reshape(geometry.shape, order="F")[span]
        super().__init__(node_numbers, geometry)


def gaussian_cfd(x: NDArrayFloat, mu: float, std: float) -> NDArrayFloat:
    return 0.5 * (1.0 + sp.special.erf((x - mu) / (std * np.sqrt(2))))


def gaussian_cfd_inv(x: NDArrayFloat, mu: float, std: float) -> NDArrayFloat:
    return sp.special.erfinv(2.0 * x - 1.0) * (std * np.sqrt(2)) + mu


def gaussian_cfd_inv_deriv(x: NDArrayFloat, mu: float, std: float) -> NDArrayFloat:
    return (
        np.sqrt(np.pi)
        * np.exp(sp.special.erfinv(2.0 * x - 1.0) ** 2)
        * (std * np.sqrt(2))
    )


class Uniform2Gaussian(Preconditioner):
    """Transform a uniform distribution into a Gaussian one."""

    def __init__(
        self,
        ud_lbound: float,
        ud_ubound: float,
        gd_mu: float,
        gd_std: float,
    ) -> None:
        """
        Initialize the instance.

        Parameters
        ----------
        ud_lbound : float
            Lower bound of the uniform distribution.
        ud_ubound : float
            Upper bound of the uniform distribution.
        gd_mu : float
            Mean of the target Gaussian distribution.
        gd_std : float
            Standard deviation of the target Gaussian distribution.
        """
        self.ud_lbound = ud_lbound
        self.ud_ubound = ud_ubound
        self.gd_mu = gd_mu
        self.gd_std = gd_std

    def _transform(self, s_raw: NDArrayFloat) -> NDArrayFloat:
        """
        Apply the preconditioning/parametrization.

        Parameters
        ----------
        s_raw : NDArrayFloat
            Non-conditioned (transformed) parameter values.

        Returns
        -------
        NDArrayFloat
            Conditioned (transformed) parameter values.
        """
        return (
            gaussian_cfd_inv(
                (s_raw - self.ud_lbound) / (self.ud_ubound - self.ud_lbound),
                mu=0.0,
                std=1.0,
            )
            * self.gd_std
            + self.gd_mu
        )

    def _backtransform(self, s_cond: NDArrayFloat) -> NDArrayFloat:
        """
        Apply the back-preconditioning/parametrization.

        Parameters
        ----------
        s_cond : NDArrayFloat
            Conditioned (transformed) parameter values.

        Returns
        -------
        NDArrayFloat
            Non-Conditioned (transformed) parameter values.
        """
        # Gaussian to uniform in several step:
        # 1) Normalize the Gaussian
        # 2) Apply the gaussian cfd to get a uniform distribution U[0, 1].
        # 3) Shift the uniform to the bounds
        return (
            gaussian_cfd(((s_cond - self.gd_mu) / self.gd_std), mu=0.0, std=1.0)
            * (self.ud_ubound - self.ud_lbound)
            + self.ud_lbound
        )

    def _dtransform_vec(
        self, s_raw: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat:
        """
        Return the transform 1st derivative times a vector as a 1-D vector..
        """
        return (
            gaussian_cfd_inv_deriv(
                (s_raw - self.ud_lbound) / (self.ud_ubound - self.ud_lbound),
                mu=0.0,
                std=1.0,
            )
            / (self.ud_ubound - self.ud_lbound)
            * self.gd_std
        ) * gradient

    def _dbacktransform_vec(
        self, s_cond: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat:
        """
        Return the backtransform 1st derivative times a vector.
        """
        return (
            (self.ud_ubound - self.ud_lbound)
            * np.exp(-((s_cond - self.gd_mu) ** 2) / (2 * self.gd_std**2))
            / (np.sqrt(2 * np.pi) * self.gd_std)
        ) * gradient

    def _dbacktransform_inv_vec(
        self, s_cond: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat:
        """
        Return the inverse of the backtransform 1st derivative times a vector.
        """
        return gradient / (
            (self.ud_ubound - self.ud_lbound)
            * np.exp(-((s_cond - self.gd_mu) ** 2) / (2 * self.gd_std**2))
            / (np.sqrt(2 * np.pi) * self.gd_std)
        )


class BoundsClipper(Preconditioner):
    """Apply an inverse absolute value preconditioning to ensure of the parameter."""

    LBOUND_RAW: float = -np.inf
    UBOUND_RAW: float = +np.inf

    def __init__(
        self,
        lbounds: Union[float, NDArrayFloat],
        ubounds: Union[float, NDArrayFloat],
    ) -> None:
        super().__init__()
        # need to store bounds for the rescaling process
        self.lbounds = lbounds
        self.ubounds = ubounds

    def _transform(self, s_raw: NDArrayFloat) -> NDArrayFloat:
        """
        Apply the preconditioning/parametrization.

        Parameters
        ----------
        s_raw : NDArrayFloat
            Non-conditioned (transformed) parameter values.

        Returns
        -------
        NDArrayFloat
            Conditioned (transformed) parameter values.
        """
        if np.any(s_raw < self.lbounds):
            raise ValueError(
                f"Found {np.count_nonzero(s_raw < self.lbounds)} "
                "values for which s_raw < lbound!"
            )
        if np.any(s_raw > self.ubounds):
            raise ValueError(
                f"Found {np.count_nonzero(s_raw > self.ubounds)} "
                "values for which s_raw > ubound!"
            )
        return s_raw

    def _backtransform(self, s_cond: NDArrayFloat) -> NDArrayFloat:
        """
        Apply the back-preconditioning/parametrization.

        Parameters
        ----------
        s_cond : NDArrayFloat
            Conditioned (transformed) parameter values.

        Returns
        -------
        NDArrayFloat
            Non-conditioned (transformed) parameter values.
        """
        return s_cond.clip(self.lbounds, self.ubounds)

    def _dtransform_vec(
        self, s_raw: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat:
        """
        Return the transform 1st derivative times a vector as a 1-D vector..
        """
        return gradient

    def _dbacktransform_vec(
        self, s_cond: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat:
        """
        Return the backtransform 1st derivative times a vector.
        """
        gradient = gradient.copy()
        # lower bound
        gradient[s_cond < self.lbounds] = 0.0
        gradient[s_cond == self.lbounds] /= 2.0
        # upper bound
        gradient[s_cond > self.ubounds] = 0.0
        gradient[s_cond == self.ubounds] /= 2.0
        return gradient

    def _dbacktransform_inv_vec(
        self, s_cond: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat:
        """
        Return the inverse of the backtransform 1st derivative times a vector.
        """
        # should have the same effect as gradient / np.sign(s_sond)
        return gradient

    def transform_bounds(self, bounds: NDArrayFloat) -> NDArrayFloat:
        """
        Transform the bounds to match the preconditioned values.

        Parameters
        ----------
        bounds : NDArrayFloat
            Array of shape (N_s, 2).

        Returns
        -------
        NDArrayFloat
            Array of shape (N_s, 2) with transformed bounds.
        """
        return bounds

    def test_preconditioner(
        self,
        lbounds: Union[float, NDArrayFloat],
        ubounds: Union[float, NDArrayFloat],
        shape: Optional[Union[int, Sequence[int]]] = None,
        rtol: float = 1e-5,
        eps: Optional[float] = None,
    ) -> None:
        """
        Test if the backconditioner and the derivatives times a vector are correct.

        This is a development tool.

        Parameters
        ----------
        lbounds : Union[float, NDArrayFloat]
            _description_
        ubounds : Union[float, NDArrayFloat]
            _description_
        shape : Optional[Union[int, Sequence[int]]], optional
            _description_, by default None
        rtol : float, optional
            _description_, by default 1e-5
        eps : Optional[float], optional
            The epsilon for the computation of the approximated preconditioner first
            derivative by finite difference. by default None.

        Raises
        ------
        ValueError
            If one of the backconditioner of the gradient conditioner are incorrect.
        """
        # Add a small epsilon to avoid boundary cases
        test_data = self._get_test_data(lbounds=lbounds, ubounds=ubounds, shape=shape)

        # 1) check by finite difference if the back-conditioner derivative is correct
        gradient = test_data.copy()
        np.testing.assert_allclose(
            self.dbacktransform_vec(test_data, gradient),
            # Finite difference differentiation
            nd.Jacobian(self.backtransform, step=eps)(test_data).T @ gradient,  # type: ignore
            rtol=rtol,
        )


def scale_pcd(scaling_factor: float, pcd: Preconditioner) -> Preconditioner:
    """Scale the given preconditioner with the scaling factor."""
    return ChainedTransforms(
        [copy.copy(pcd), LinearTransform(slope=scaling_factor, y_intercept=0.0)]
    )


def get_max_update(
    pcd: Preconditioner,
    s_nc: NDArrayFloat,
    grad_nc: NDArrayFloat,
    is_preconditioned: bool,
) -> float:
    """
    Get the max update of parameter values with gradient descent on conditioned values.

    Parameters
    ----------
    pcd : Preconditioner
        Preconditioner.
    s_nc : NDArrayFloat
        Non conditioned parameter values.
    grad_nc : NDArrayFloat
        Gradient of the objective function with respect to the non conditioned parameter
        values.
    is_preconditioned: bool
        Whether the max update is evaluated on the preconditioned values or not.

    Returns
    -------
    float
        Maximum update of the parameter values.
    """
    pcd = copy.copy(pcd)
    s_cond = pcd(s_nc)
    if is_preconditioned:
        return float(
            sp.linalg.norm(
                pcd.dbacktransform_vec(s_cond, grad_nc),
                ord=np.inf,
            )
        )
    else:
        return float(
            sp.linalg.norm(
                pcd.backtransform(s_cond - pcd.dbacktransform_vec(s_cond, grad_nc))
                - s_nc,
                ord=np.inf,
            )
        )


def get_max_update_wrapper(
    scaling_factor: float,
    pcd: Preconditioner,
    s_nc: NDArrayFloat,
    grad_nc: NDArrayFloat,
    is_preconditioned: bool,
) -> float:
    """
    Get the max update of parameter values with gradient descent on conditioned values.

    Parameters
    ----------
    pcd : Preconditioner
        Preconditioner.
    s_nc : NDArrayFloat
        Non conditioned parameter values.
    grad_nc : NDArrayFloat
        Gradient of the objective function with respect to the non conditioned parameter
        values.
    is_preconditioned: bool
        Whether the max update is evaluated on the preconditioned values or not.

    Returns
    -------
    float
        Maximum update of the parameter values.
    """
    return get_max_update(
        scale_pcd(scaling_factor, pcd), s_nc, grad_nc, is_preconditioned
    )


def cost_fun(
    scaling_factor: float,
    pcd: Preconditioner,
    s_nc: NDArrayFloat,
    grad_nc: NDArrayFloat,
    max_update_target,
    is_target_preconditioned: bool,
) -> float:
    return np.log(
        (
            get_max_update(
                scale_pcd(scaling_factor, pcd),
                s_nc=s_nc,
                grad_nc=grad_nc,
                is_preconditioned=is_target_preconditioned,
            )
            - max_update_target
        )
        ** 2
        + 1.0
    )  # Add 1.0 because of the log


def get_relative_error(x: float, x_ref: float) -> float:
    """
    Get the relative error between x and the reference x_ref.

    Parameters
    ----------
    x : float
        Value.
    x_ref : float
        Reference value.

    Returns
    -------
    float
        Relative error between x and x_ref.
    """
    return (x - x_ref) / x_ref


def is_picklable(obj):
    try:
        pickle.dumps(obj)

    except (pickle.PicklingError, TypeError):
        return False
    return True


@dataclass
class GradientScalerConfig:
    """
    Configuration for the gradient scaling when using L-BFGS-B for the inversion.

    Attributes
    ----------
    max_update_target : float
        Maximum update desired on the parameter values.
    is_target_preconditioned : bool
        Whether the target is defined for the preconditioned parameter values or not.
    max_workers : int, optional
        The maximum number of workers to evaluate the maximum change in parallel. If the
        preconditioner to not picklable, then it is set to 1. By default 50.
    rtol : float, optional
        Relative tolerance on the target to consider a convergence. By default 0.05.
    lb : float, optional
        Lower bound for the searched interval in first round. By default 1e-10.
    ub : float, optional
        Upper bound for the searched interval in first round. By default 1e10.
    n_samples_in_first_round : int, optional
        Number of samples used to cover the searched interval in the first round.
        By default 50.
    """

    max_update_target: float
    is_target_preconditioned: bool
    max_workers: int = 50
    rtol: float = 0.05
    lb: float = 1e-10
    ub: float = 1e10
    n_samples_in_first_round: int = 50


def scale_preconditioned_gradient(
    s_nc: NDArrayFloat,
    grad_nc: NDArrayFloat,
    pcd: Preconditioner,
    gsc: GradientScalerConfig,
    logger: Optional[logging.Logger] = None,
) -> Preconditioner:
    """
    Add a LinearTransform to the precondition gradient to ensure a defined update.

    TODO: add the maths and explanations.

    Parameters
    ----------
    s_nc : NDArrayFloat
        Non conditioned parameter values.
    grad_cond : NDArrayFloat
        Conditioned gradient.
    pcd : Preconditioner
        Preconditioner instance.
    gsc: GradientScalerConfig
        Configuration for the gradient scaling.
    logger : Optional[logging.Logger], optional
        Optional :class:`logging.Logger` instance used for event logging.
        The default is None.

    Returns
    -------
    Preconditioner
        The updated preconditioner with the linear transform.
    """
    if logger is not None:
        logger.info("Scaling the preconditioned gradient!")
        init_max_change = get_max_update(
            copy.copy(pcd), s_nc, grad_nc, gsc.is_target_preconditioned
        )
        logger.info("Initial scaling factor = 1.0")
        logger.info(f"Initial maximum change   = {init_max_change:.2e}")
        logger.info(f"Objective maximum change = {gsc.max_update_target:.2e}\n")

    # If the initial maximum change already respects the objective, then leave
    if (
        np.abs(
            get_relative_error(
                init_max_change,
                gsc.max_update_target,
            )
        )
        <= gsc.rtol
    ):
        if logger is not None:
            logger.info("Target already fulfilled, skipping optimization \n")
        return pcd

    # step 1: explore
    scaling_factor = 1.0  # initial guess
    round = 1

    def get_pcd() -> Generator:
        while True:
            yield copy.copy(pcd)

    def get_s_nc() -> Generator:
        while True:
            yield s_nc

    def get_grad_nc() -> Generator:
        while True:
            yield grad_nc

    def get_is_target_preconditioned() -> Generator:
        while True:
            yield gsc.is_target_preconditioned

    _max_workers = 1
    if is_picklable(pcd) and gsc.max_workers != 1:
        _max_workers = gsc.max_workers

    # minimum 50 samples
    if gsc.n_samples_in_first_round < 50:
        if logger is not None:
            logger.info("Setting 'samples_in_first_round' to 50!\n")
        n_samples_in_first_round = 50
    else:
        n_samples_in_first_round = gsc.n_samples_in_first_round

    lb = copy.copy(gsc.lb)
    ub = copy.copy(gsc.ub)

    while (
        np.abs(
            get_relative_error(
                get_max_update(
                    scale_pcd(scaling_factor, pcd),
                    s_nc,
                    grad_nc,
                    gsc.is_target_preconditioned,
                ),
                gsc.max_update_target,
            )
        )
        > gsc.rtol
    ):
        if round == 6:
            if logger is not None:
                logger.info(
                    "Did not converge in 5 rounds! The update target might"
                    " not be feasible for the given preconditioner >>"
                    "The scaling factor remains 1."
                )
            return pcd

        if logger is not None:
            logger.info(f"Optimization round {round}")
            logger.info(f"lower bound   = {lb:.2e}")
            logger.info(f"upper bound   = {ub:.2e}")

        scaling_factors = np.logspace(
            np.log10(lb),
            np.log10(ub),
            n_samples_in_first_round - (round - 1) * 10,
            base=10,
        )

        if _max_workers == 1:
            max_s_nc_updates: List[float] = []
            for _scaling_factor in scaling_factors:
                max_s_nc_updates.append(
                    get_max_update(
                        scale_pcd(_scaling_factor, copy.copy(pcd)),
                        s_nc,
                        grad_nc,
                        gsc.is_target_preconditioned,
                    )
                )
        else:
            with ProcessPoolExecutor(max_workers=_max_workers) as executor:
                max_s_nc_updates = list(
                    executor.map(
                        get_max_update_wrapper,
                        scaling_factors,
                        get_pcd(),
                        get_s_nc(),
                        get_grad_nc(),
                        get_is_target_preconditioned(),
                    )
                )

        squared_diff = (np.array(max_s_nc_updates) - gsc.max_update_target) ** 2
        argmin = np.argmin(np.log(squared_diff + 1.0))
        scaling_factor = scaling_factors[argmin]
        if argmin == 0:
            lb = scaling_factor
        else:
            lb = scaling_factors[argmin - 1]
        if argmin == len(scaling_factors) - 1:
            ub = scaling_factor
        else:
            ub = scaling_factors[argmin + 1]

        if logger is not None:
            logger.info(f"Post round {round}: Scaling factor  = {scaling_factor:.2e}")
            logger.info(
                f"Post round {round}: Max s_nc change = {max_s_nc_updates[argmin]:.2e}"
            )
            _re = get_relative_error(max_s_nc_updates[argmin], gsc.max_update_target)
            logger.info(f"Post round {round}: Rel. error to target = {_re:.2%}\n")

        # update the round number
        round += 1

    return ChainedTransforms(
        [copy.copy(pcd), LinearTransform(slope=scaling_factor, y_intercept=0.0)]
    )
