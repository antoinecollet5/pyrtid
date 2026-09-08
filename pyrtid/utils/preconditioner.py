# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 Antoine COLLET

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

from __future__ import annotations

import copy
import logging
import pickle
import warnings
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Generator, List, Optional, Sequence, Tuple, Union

import covmats
import numdifftools as nd
import numpy as np
import scipy as sp
from scipy.sparse.linalg import LinearOperator

from pyrtid.utils import (
    NDArrayBool,
    NDArrayFloat,
    NDArrayInt,
    RectilinearGrid,
    check_random_state,
    object_or_object_sequence_to_list,
)


class Preconditioner(ABC):
    """
    This is an abstract class for parameter preconditioning and parametrization.

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
            raise ValueError("'transform' method expects a 1D vector!")
        self.test_bounds_tr(s_raw)  # test that s_raw is in the supported range
        # call the _transform method defined in child classes
        return self._transform(s_raw)

    def backtransform(self, s_cond: NDArrayFloat) -> NDArrayFloat:
        """
        Apply the back-preconditioning/parametrization.

        Parameters
        ----------
        s_cond : NDArrayFloat
            The conditioned values as a 1D array.

        Returns
        -------
        NDArrayFloat
            The non-conditioned values as a 1D vector.
        """
        if not s_cond.ndim == 1:
            raise ValueError("'backtransform' method expects a 1D vector!")
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
        Return the transform 1st derivative times a vector as a 1-D vector.

        Because the preconditioner operates a variable change in the function: the new
        objective function J is J2(s2) = J[s], with s the adjusted parameter vector.
        Then the gradient is dJ/ds = ds2[s]/ds * dJ2[s2]/ds2. This is useful when the
        derivative is computed on s2 but is required w.r.t s.

        - Example 1 : we defined s2[s] = k * s
          -> dJ/ds = ds2[s]/ds dJ2[s2]/ds2 = k * dJ2/ds2. If
          k = 1/100 --> the gradient dJ[s]/ds is a 100 times weaker than dJ2/ds2.
        - Example 2: we defined s2 = log(s)
          -> dJ/ds = ds2[s]/ds dJ2[s2]/ds2 = 1/s * dJ2[s2]/ds2
          -> dJ/ds * s = dJ2[s2]/ds2


        Often, it is more efficient to compute (ds2/ds * dJ2/ds2) directly
        than to return ds2/ds, especially if ds2/ds is a matrix of large dimension.

        Parameters
        ----------
        s_raw : NDArrayFloat
            The non-conditioned values with size $N_{s}$ at which the derivative
            is evaluated.
        gradient : NDArrayFloat
            Any vector with size $N_{s2}$, typically the gradient of the objective
            function w.r.t. the conditioned values.

        Returns
        -------
        NDArrayFloat
            Product of the 1st derivative w.r.t. the conditioned (transformed)
            values and any vector b with size $N_{s2}$.
        """
        if not s_raw.ndim == 1 or not gradient.ndim == 1:
            raise ValueError("'dtransform_vec' method expects 1D vectors!")
        self.test_bounds_tr(s_raw)  # test that s_raw is in the supported range
        # call the _dtransform_vec method defined in child classes
        return self._dtransform_vec(s_raw, gradient)

    @abstractmethod
    def _dtransform_vec(
        self, s_raw: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat:
        """Return the transform 1st derivative times a vector."""
        ...  # pragma: no cover

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
        than to return ds/ds2, especially if ds/ds2 is a matrix of large dimension.

        Parameters
        ----------
        s_cond : NDArrayFloat
            The conditioned values with size $N_{s2}$ at which the derivative
            is evaluated.
        gradient : NDArrayFloat
            Any vector with size $N_{s}$, typically the gradient of the objective
            function w.r.t. the non-conditioned values.

        Returns
        -------
        NDArrayFloat
            Product of the 1st derivative w.r.t. the conditioned (transformed)
            values and any vector b with size $N_{s}$.
        """
        if not s_cond.ndim == 1 or not gradient.ndim == 1:
            raise ValueError("'dbacktransform_vec' method expects 1D vectors!")

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
        than to return ds/ds2, especially if ds/ds2 is a matrix of large dimension.

        Parameters
        ----------
        s_cond : NDArrayFloat
            The conditioned values with size $N_{s2}$ at which the derivative
            is evaluated.
        gradient : NDArrayFloat
            Any vector with size $N_{s}$, typically the result of
            :meth:`dbacktransform_vec` that we want to invert.

        Returns
        -------
        NDArrayFloat
            Product of the 1st derivative w.r.t. the conditioned (transformed)
            values and any vector b with size $N_{s}$.

        Raises
        ------
        NotImplementedError
            If the given preconditioner does not support this operation (e.g.
            because the backtransform derivative is not invertible).
        """
        if not s_cond.ndim == 1 or not gradient.ndim == 1:
            raise ValueError("'dbacktransform_inv_vec' method expects 1D vectors!")

        self.test_bounds_btr(s_cond)  # test that s_cond is in the supported range
        # call the _dbacktransform_vec method defined in child classes
        return self._dbacktransform_inv_vec(s_cond, gradient)

    @abstractmethod
    def _dbacktransform_vec(
        self, s_cond: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat:
        """Return the backtransform 1st derivative times a vector."""
        ...  # pragma: no cover

    @abstractmethod
    def _dbacktransform_inv_vec(
        self, s_cond: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat:
        """Return the inverse of the backtransform 1st derivative times a vector."""
        ...  # pragma: no cover

    def __call__(self, s_raw: NDArrayFloat) -> NDArrayFloat:
        """Call the preconditioner (alias for :meth:`transform`)."""
        return self.transform(s_raw)

    def _get_test_data(
        self,
        lbounds: Union[float, NDArrayFloat],
        ubounds: Union[float, NDArrayFloat],
        shape: Optional[Union[int, Sequence[int]]] = None,
    ) -> NDArrayFloat:
        """
        Get test data to check the preconditioner correctness.

        This is a development tool. It draws values uniformly within
        ``[lbounds, ubounds]`` (infinite bounds are clipped to +/-1e10 so that
        sampling remains well defined).

        Parameters
        ----------
        lbounds : Union[float, NDArrayFloat]
            Lower bound(s) of the range from which test values are drawn.
        ubounds : Union[float, NDArrayFloat]
            Upper bound(s) of the range from which test values are drawn.
        shape : Optional[Union[int, Sequence[int]]], optional
            Shape of the returned test array. If None, it is inferred from
            the size of `lbounds`/`ubounds` (defaulting to 50 samples for
            scalar bounds). The default is None.

        Returns
        -------
        NDArrayFloat
            Array of test values uniformly sampled within the given bounds.
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
            Non-conditioned values.

        Raises
        ------
        ValueError
            If the provided values are out of the supported range.
        """
        if np.any(np.logical_or(self.LBOUND_RAW > s_raw, self.UBOUND_RAW < s_raw)):
            raise ValueError(
                "The provided parameter min and max "
                f"({np.min(s_raw)}, {np.max(s_raw)}) "
                "do not match with the "
                "range of values supported by the transform: "
                f"[{self.LBOUND_RAW}, {self.UBOUND_RAW}]"
            )

    def test_bounds_btr(self, s_cond: NDArrayFloat) -> None:
        """
        Test the bounds for back-transformation.

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
                "The provided parameter values min and max "
                f"({np.min(s_cond)}, {np.max(s_cond)}) "
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

        This is a development tool: it checks (1) that `backtransform` is the
        inverse of `transform`, (2) that `dtransform_vec` matches a finite
        difference approximation, (3) that `dbacktransform_vec` matches a
        finite difference approximation, and (4) that `dbacktransform_inv_vec`
        correctly inverts `dbacktransform_vec` (skipped if the preconditioner
        raises `NotImplementedError` for that operation).

        Parameters
        ----------
        lbounds : Union[float, NDArrayFloat]
            Lower bound(s) used to generate random test values.
        ubounds : Union[float, NDArrayFloat]
            Upper bound(s) used to generate random test values.
        shape : Optional[Union[int, Sequence[int]]], optional
            Shape of the generated test values. The default is None.
        rtol : float, optional
            Relative tolerance used for all correctness checks.
            The default is 1e-5.
        eps : Optional[float], optional
            The epsilon for the computation of the approximated preconditioner first
            derivative by finite difference. The default is None.

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
            Lower bound(s) used to generate random test values.
        ubounds : Union[float, NDArrayFloat]
            Upper bound(s) used to generate random test values.
        shape : Optional[Union[int, Sequence[int]]], optional
            Shape of the generated test values. The default is None.
        rtol : float, optional
            Relative tolerance used for all correctness checks.
            The default is 1e-5.
        eps : Optional[float], optional
            The epsilon for the computation of the approximated preconditioner first
            derivative by finite difference. The default is None.
        skip_checks: Optional[Sequence[int]]
            List of checks to skip (1: backtransform/transform inverse check,
            2: dtransform_vec finite-difference check, 3: dbacktransform_vec
            finite-difference check, 4: dbacktransform_inv_vec check). This is
            useful when some preconditioner will fail tests while remaining
            correct. The default is None.

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
                @ gradient,
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

    def smart_copy(self) -> Preconditioner:
        """
        Return a copy of the instance mixing shallow and deep copy.

        Since some preconditioners are modified inplace when calling :func:`transform`
        or :func:`backtransform`, it is necessary to deepcopy some attributes so that
        the original instance is not altered.
        """
        return copy.copy(self)


class ChainedTransforms(Preconditioner):
    """Combination of multiple preconditioners applied one after the other."""

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
        out = s_raw
        # successively transform the data with each preconditioner
        for pcd in self.pcds:
            out = pcd.transform(out)
        return out

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
        out = s_cond
        # successively backtransform the data with each preconditioner
        # in the reversed order
        for pcd in reversed(self.pcds):
            out = pcd.backtransform(out)
        return out

    def _dtransform_vec(
        self, s_raw: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat:
        """Return the transform 1st derivative times a vector as a 1-D vector."""
        s, _gradient = s_raw, gradient
        for pcd in self.pcds:
            _gradient = pcd.dtransform_vec(s, _gradient)
            s = pcd(s)
        return _gradient

    def _dbacktransform_vec(
        self, s_cond: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat:
        """Return the backtransform 1st derivative times a vector."""
        # dJ3/ds3 = ds2[s3]/ds3 * ds[s2]/ds2 * dJ/ds ...
        s, _gradient = self.backtransform(s_cond), gradient
        for pcd in self.pcds:
            _gradient = pcd.dbacktransform_vec(pcd(s), _gradient)
            s = pcd(s)
        return _gradient

    def _dbacktransform_inv_vec(
        self, s_cond: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat:
        """Return the inverse of the backtransform 1st derivative times a vector."""
        s, _gradient = s_cond, gradient
        # successively backtransform the data with each preconditioner
        # in the reversed order
        for pcd in reversed(self.pcds):
            _gradient = pcd.dbacktransform_inv_vec(s, _gradient)
            s = pcd.backtransform(s)
        return _gradient

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
        out = bounds
        for pcd in self.pcds:
            out = pcd.transform_bounds(out)
        return out

    def smart_copy(self) -> ChainedTransforms:
        """Return a copy of the instance, deep-copying each chained preconditioner."""
        return ChainedTransforms([p.smart_copy() for p in self.pcds])


class NoTransform(Preconditioner):
    """Does not apply any preconditioning (identity transform)."""

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
        """Return the transform 1st derivative times a vector as a 1-D vector."""
        return gradient

    def _dbacktransform_vec(
        self, s_cond: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat:
        """Return the backtransform 1st derivative times a vector."""
        return gradient

    def _dbacktransform_inv_vec(
        self, s_cond: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat:
        """Return the inverse of the backtransform 1st derivative times a vector."""
        return gradient


class LinearTransform(Preconditioner):
    """Apply a linear transform to the parameter (``s_cond = slope * s_raw + b``)."""

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
        """Return the transform 1st derivative times a vector as a 1-D vector."""
        return self.slope * gradient

    def _dbacktransform_vec(
        self, s_cond: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat:
        """Return the backtransform 1st derivative times a vector."""
        return 1 / self.slope * gradient

    def _dbacktransform_inv_vec(
        self, s_cond: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat:
        """Return the inverse of the backtransform 1st derivative times a vector."""
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
        """Return the transform 1st derivative times a vector as a 1-D vector."""
        return 1 / (2.0 * np.sqrt(s_raw)) * gradient

    def _dbacktransform_vec(
        self, s_cond: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat:
        """Return the backtransform 1st derivative times a vector."""
        return 2.0 * s_cond * gradient

    def _dbacktransform_inv_vec(
        self, s_cond: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat:
        """Return the inverse of the backtransform 1st derivative times a vector."""
        return gradient / 2.0 / s_cond


class InvAbsTransform(Preconditioner):
    """Apply an inverse absolute value preconditioning (tracks and restores sign)."""

    LBOUND_RAW: float = 0.0
    UBOUND_RAW: float = +np.inf

    def __init__(self) -> None:
        """Initialize the instance, with all signs initially positive."""
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
        """Return the transform 1st derivative times a vector as a 1-D vector."""
        return self.signs * gradient

    def _dbacktransform_vec(
        self, s_cond: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat:
        """Return the backtransform 1st derivative times a vector."""
        return np.sign(s_cond) * gradient

    def _dbacktransform_inv_vec(
        self, s_cond: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat:
        """Return the inverse of the backtransform 1st derivative times a vector."""
        # should have the same effect as gradient / np.sign(s_cond)
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
            Array of shape (N_s, 2) with transformed bounds. Unchanged here since
            the sign is data-dependent and cannot be inferred from static bounds.
        """
        return bounds


class LogTransform(Preconditioner):
    """Apply a log preconditioning to ensure positive values of the parameter."""

    LBOUND_RAW = 1e-100  # cannot be zero
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
        """Return the transform 1st derivative times a vector as a 1-D vector."""
        return 1 / s_raw * gradient

    def _dbacktransform_vec(
        self, s_cond: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat:
        """Return the backtransform 1st derivative times a vector."""
        return np.exp(s_cond) * gradient

    def _dbacktransform_inv_vec(
        self, s_cond: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat:
        """Return the inverse of the backtransform 1st derivative times a vector."""
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
        Value of the function's midpoint, by default 0.0
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
        Value of the function's midpoint, by default 0.0
    rate : float, optional
        The logistic growth rate or steepness of the curve, by default 1.0
    supremum : float, optional
        The supremum of the values of the function, by default 1.0

    Returns
    -------
    NDArrayFloat
        Logit values.
    """
    return -np.log(supremum / s - 1.0) / rate + s0


def tanh_wrapper(
    s: NDArrayFloat, s0: float = 0.0, rate: float = 1.0, supremum: float = 1.0
) -> NDArrayFloat:
    """
    Return the hyperbolic tangent function (inverse to arctanh_wrapper).

    Parameters
    ----------
    s : NDArrayFloat
        Input values.
    s0 : float, optional
        Value of the function's midpoint, by default 0.0
    rate : float, optional
        The logistic growth rate or steepness of the curve, by default 1.0
    supremum : float, optional
        The supremum of the values of the function, by default 1.0

    Returns
    -------
    NDArrayFloat
        Hyperbolic tangent values, rescaled to ``[0, supremum]``.
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
        Value of the function's midpoint, by default 0.0
    rate : float, optional
        The logistic growth rate or steepness of the curve, by default 1.0
    supremum : float, optional
        The supremum of the values of the function, by default 1.0

    Returns
    -------
    NDArrayFloat
        Derivative of :func:`tanh_wrapper` w.r.t. ``s``.
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
        Input values, expected in ``[0, supremum]``.
    s0 : float, optional
        Value of the function's midpoint, by default 0.0
    rate : float, optional
        The logistic growth rate or steepness of the curve, by default 1.0
    supremum : float, optional
        The supremum of the values of the function, by default 1.0

    Returns
    -------
    NDArrayFloat
        Inverse hyperbolic tangent values.
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
        Input values, expected in ``[0, supremum]``.
    s0 : float, optional
        Value of the function's midpoint, by default 0.0
    rate : float, optional
        The logistic growth rate or steepness of the curve, by default 1.0
    supremum : float, optional
        The supremum of the values of the function, by default 1.0

    Returns
    -------
    NDArrayFloat
        Derivative of :func:`arctanh_wrapper` w.r.t. ``s``.
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
    is_log10: bool
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
    Return the derivative (w.r.t. s_raw) of :func:`to_new_range`.

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
    is_log10: bool
        Whether to use a log10 scale for the new range.

    Returns
    -------
    NDArrayFloat
        Derivative of the rescaled output values w.r.t. `s_raw`.
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
        old_lbound : float
            Lower bound of the original (non-conditioned) values.
        old_ubound : float
            Upper bound of the original (non-conditioned) values.
        new_lbound : float
            Lower bound of the conditioned (rescaled) values. This does not
            really matter, e.g. -5.0.
        new_ubound : float
            Upper bound of the conditioned (rescaled) values. This does not
            really matter, e.g. 5.0.
        is_log10: bool
            Whether to use a log10-scaling for the rescaling.
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
        # rescaling between new_lbound and new_ubound -> the scale does not
        # really matter
        return to_new_range(
            np.log10(s_raw) if self.is_log10 else s_raw,
            old_lbound=(
                np.log10(self.old_lbound) if self.is_log10 else self.old_lbound
            ),
            old_ubound=(
                np.log10(self.old_ubound) if self.is_log10 else self.old_ubound
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
        """Return the transform 1st derivative times a vector as a 1-D vector."""
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
        """Return the backtransform 1st derivative times a vector."""
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
        """Return the inverse of the backtransform 1st derivative times a vector."""
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
    in the interval ``]0, supremum[``.

    This parametrization is particularly useful when a parameter
    has two modes that are the limits of the value range. For example,
    to image a porosity field with two facies, one porous, the other not,
    with more or less homogeneous values for each facies (e.g. 15% and 40%).
    """

    def __init__(
        self, s0: float = 0.0, rate: float = 1.0, supremum: float = 1.0
    ) -> None:
        """
        Initialize the instance.

        Parameters
        ----------
        s0: float
            Value of the function's midpoint. The default is 0.0.
        rate: float
            Growth rate. The higher the rate, the steeper the sigmoid. A value of
            3 is usually the upper acceptable limit, i.e., above this value,
            the bijection between "transform" and "backtransform" might be lost and the
            derivative might become incorrect. The default is 1.0.
        supremum: float
            The supremum of the values supported by the transform, i.e., the raw
            values must lie in ``]0, supremum[``. The default is 1.0.
        """
        self.s0: float = s0
        self.rate: float = rate
        self.supremum: float = supremum
        self.LBOUND_RAW = 0.0
        self.UBOUND_RAW = supremum

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
        """Return the transform 1st derivative times a vector as a 1-D vector."""
        return (
            darctanh_wrapper(s_raw, s0=self.s0, rate=self.rate, supremum=self.supremum)
            * gradient
        )

    def _dbacktransform_vec(
        self, s_cond: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat:
        """Return the backtransform 1st derivative times a vector."""
        # otherwise chain rule of two functions
        return (
            dtanh_wrapper(s_cond, s0=self.s0, rate=self.rate, supremum=self.supremum)
            * gradient
        )

    def _dbacktransform_inv_vec(
        self, s_cond: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat:
        """Return the inverse of the backtransform 1st derivative times a vector."""
        return gradient / (
            dtanh_wrapper(s_cond, s0=self.s0, rate=self.rate, supremum=self.supremum)
        )


class SigmoidRescalerBounded(ChainedTransforms):
    """
    Chain a :class:`RangeRescaler` (to ``]0, 1[``) with
    a :class:`SigmoidRescaler`.
    """

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
        old_lbound : float
            Lower bound of the original values.
        old_ubound : float
            Upper bound of the original values.
        rate: float
            Growth rate. The higher the rate, the steeper the sigmoid. A value of
            3 is usually the upper acceptable limit, i.e., above this value,
            the bijection between "transform" and "backtransform" might be lost and the
            derivative might become incorrect.
        is_log10: bool
            Whether to use a log10-scaling for the logit y-scale.
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
            Array of shape (N_s, 2), constant and set to
            ``[LBOUND_COND, UBOUND_COND]`` since the sigmoid never actually
            reaches its asymptotes.
        """
        return np.array(
            [
                np.ones_like(bounds[:, 0]) * self.LBOUND_COND,
                np.ones_like(bounds[:, 0]) * self.UBOUND_COND,
            ]
        ).T


class Normalizer(Preconditioner):
    """Center and scale the parameter by the prior field's mean and std deviation."""

    def __init__(self, s_prior: NDArrayFloat) -> None:
        """
        Initialize the instance.

        Parameters
        ----------
        s_prior : NDArrayFloat
            Prior field used to compute the (scalar) mean and standard
            deviation used for the normalization.
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
        """Return the transform 1st derivative times a vector as a 1-D vector."""
        return gradient / self.prior_std

    def _dbacktransform_vec(
        self, s_cond: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat:
        """Return the backtransform 1st derivative times a vector."""
        return self.prior_std * gradient

    def _dbacktransform_inv_vec(
        self, s_cond: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat:
        """Return the inverse of the backtransform 1st derivative times a vector."""
        return gradient / self.prior_std


class StdRescaler(Preconditioner):
    """Rescale the deviation of the parameter from a (possibly spatial) prior field."""

    def __init__(
        self,
        s_prior: NDArrayFloat,
        prior_std: Optional[float] = None,
    ) -> None:
        """
        Initialize the instance.

        Parameters
        ----------
        s_prior : NDArrayFloat
            Prior field, subtracted from/added back to the raw values.
        prior_std : Optional[float], optional
            Standard deviation used for the rescaling. If None, it is
            computed from `s_prior`. The default is None.
        """
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
        """Return the transform 1st derivative times a vector as a 1-D vector."""
        return gradient / self.prior_std

    def _dbacktransform_vec(
        self, s_cond: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat:
        """Return the backtransform 1st derivative times a vector."""
        return self.prior_std * gradient

    def _dbacktransform_inv_vec(
        self, s_cond: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat:
        """Return the inverse of the backtransform 1st derivative times a vector."""
        return gradient / self.prior_std


class BoundsRescaler(Preconditioner):
    """Apply a bound rescaling (aka logit)."""

    EPSILON = 1e-10

    def __init__(
        self,
        lbounds: Union[float, NDArrayFloat],
        ubounds: Union[float, NDArrayFloat],
    ) -> None:
        """
        Initialize the instance.

        Parameters
        ----------
        lbounds : Union[float, NDArrayFloat]
            Lower bound(s) of the non-conditioned values.
        ubounds : Union[float, NDArrayFloat]
            Upper bound(s) of the non-conditioned values.
        """
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
        """Return the transform 1st derivative times a vector as a 1-D vector."""
        _s_raw = s_raw.clip(self.lbounds + self.EPSILON, self.ubounds - self.EPSILON)
        return (
            -(self.ubounds - self.lbounds)
            / ((_s_raw - self.lbounds) * (_s_raw - self.ubounds))
            * gradient
        )

    def _dbacktransform_vec(
        self, s_cond: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat:
        """Return the backtransform 1st derivative times a vector."""
        return (
            (self.ubounds - self.lbounds) * np.exp(s_cond) / (np.exp(s_cond) + 1) ** 2
        ) * gradient

    def _dbacktransform_inv_vec(
        self, s_cond: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat:
        """Return the inverse of the backtransform 1st derivative times a vector."""
        return gradient / (
            (self.ubounds - self.lbounds) * np.exp(s_cond) / (np.exp(s_cond) + 1) ** 2
        )


def get_gd_weights(theta: NDArrayFloat) -> NDArrayFloat:
    """
    Return the Gradual Deformation (GD) weights associated with `theta`.

    The weights are built such that the sum of their squares equals one, so
    that the linear combination of white noises they define
    (see :func:`gd_parametrize`) remains a standard-normal white noise.

    Parameters
    ----------
    theta : NDArrayFloat
        Gradual deformation parameter vector, with size ``Ne - 1`` where
        ``Ne`` is the number of combined realizations/white noises.

    Returns
    -------
    NDArrayFloat
        GD weights, with size ``Ne``.
    """
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
    W : NDArrayFloat
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
    b : NDArrayFloat
        Vector with size (N_z,) to multiply the derivative with, typically the
        gradient w.r.t. the GD-parametrized field ``Z = gd_parametrize(z_arr,
        get_gd_weights(theta))``.

    Returns
    -------
    NDArrayFloat
        Product of the derivative of ``Z`` w.r.t. `theta` and `b`, with size
        equal to ``np.size(theta)``.
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
    """
    Validate and cast the number of realizations `ne` used in GD parametrizations.

    Parameters
    ----------
    ne : int
        Candidate number of realizations. Must be castable to an integer
        that is greater than or equal to 2.

    Returns
    -------
    int
        The validated number of realizations.

    Raises
    ------
    ValueError
        If `ne` cannot be cast to an integer, or if it is lower than 2.
    """
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
    Apply a Gradual Deformation parametrization for a non-conditional field.

    The Gradual Deformation (GD) parametrization generates a white noise
    (reduced and centered) as a linear combination of Ne white noises while
    adjusting Ne-1 parameters (``theta``). The obtained white noise is
    "colorized" through a :class:`covmats.CovarianceMatrix` (``cov``) to
    generate a field with the desired geostatistical covariance -- any
    ``covmats`` backend works here (dense/sparse Cholesky, sparse precision
    Cholesky, ensemble, eigen-factorized, ...), which generalizes the former
    SPDE-only implementation.

    Here: non-conditional simulation (see :class:`GDPCS` for the conditional
    counterpart).

    Notes
    -----
    :meth:`_dbacktransform_vec` requires ``cov.colorize_adjoint``, which is
    available on all dense/eigen/ensemble ``covmats`` backends but not yet
    on the sparse ones (see the module docstring); it raises
    ``NotImplementedError`` in that case. :meth:`_dbacktransform_inv_vec` is
    not implemented (not invertible), as in the previous SPDE-based version.

    References
    ----------
    Hu, L.Y. (2000). "Gradual Deformation and Iterative Calibration of
    Gaussian-Related Stochastic Models." Mathematical Geology 32, 87-108.
    """

    def __init__(
        self,
        ne: int,
        cov: covmats.CovarianceMatrix,
        estimated_mean: float = 0.0,
        theta: Optional[NDArrayFloat] = None,
        random_state: Optional[
            Union[int, np.random.Generator, np.random.RandomState]
        ] = None,
        is_update_mean: bool = True,
    ) -> None:
        """
        Initialize the instance.

        Parameters
        ----------
        ne : int
            Number of independent white noise realizations combined by the
            gradual deformation.
        cov : covmats.CovarianceMatrix
            Covariance representation of the field to simulate, with shape
            ``(n, n)``. Only ``cov.colorize`` is required for the forward
            (non-conditional) simulation.
        estimated_mean : float, optional
            Initial estimated (constant) mean of the field. The default is 0.0.
        theta : Optional[NDArrayFloat], optional
            Initial gradual deformation parameter, with size `ne - 1`. If
            None, it is initialized so that all Ne realizations are equally
            weighted (see :func:`get_theta_init_uniform`). The default is None.
        random_state : Optional[Union[int, np.random.Generator, np.random.RandomState]]
            Pseudorandom number generator state used to draw the ensemble of
            `ne` white noises. If `random_state` is ``None`` (or `np.random`),
            the `numpy.random.RandomState` singleton is used.
            If `random_state` is an int, a new ``RandomState`` instance is used,
            seeded with `random_state`.
            If `random_state` is already a ``Generator`` or ``RandomState``
            instance then that instance is used. The default is None
        is_update_mean : bool, optional
            Whether the (constant) field mean is also adjusted alongside
            `theta`. The default is True.
        """
        # initialize the super instance
        super().__init__()

        self.estimated_mean: float = estimated_mean
        self.is_update_mean: bool = is_update_mean
        self.cov: covmats.CovarianceMatrix = cov

        # Dimension of the white noise expected by `cov.colorize`: the full
        # field size for full-rank representations, or the retained rank
        # (`subspace_size`) for low-rank ones (e.g. CovViaEnsemble).
        self._colorize_dim: int = int(
            getattr(cov, "subspace_size", None) or cov.shape[0]
        )

        # initialize the ensemble of white noises with shape (colorize_dim, ne)
        self.W: NDArrayFloat = check_random_state(random_state).normal(
            size=(self._colorize_dim, ne)
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

        This is a development tool. Unlike the base class implementation, it
        does not sample uniformly within `lbounds`/`ubounds` (those are
        ignored): it back-transforms the current `theta`/`estimated_mean`,
        since arbitrary values are not valid GD-parametrized fields.

        Parameters
        ----------
        lbounds : Union[float, NDArrayFloat]
            Ignored, kept for interface compatibility with the base class.
        ubounds : Union[float, NDArrayFloat]
            Ignored, kept for interface compatibility with the base class.
        shape : Optional[Union[int, Sequence[int]]], optional
            Ignored, kept for interface compatibility with the base class.

        Returns
        -------
        NDArrayFloat
            The field obtained by back-transforming the current `theta`
            (and `estimated_mean` if `is_update_mean` is True).
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
        assert s_raw.size == self.cov.shape[0]
        if self.is_update_mean:
            return np.hstack((self.theta, self.estimated_mean))
        return self.theta

    def _backtransform(self, s_cond: NDArrayFloat) -> NDArrayFloat:
        """
        Apply the back-preconditioning/parametrization.

        Parameters
        ----------
        s_cond : NDArrayFloat
            Conditioned (transformed) parameter values: the GD parameter
            `theta` (plus `estimated_mean` if `is_update_mean` is True).

        Returns
        -------
        NDArrayFloat
            Non-conditioned (transformed) parameter values, i.e. the
            GD-parametrized, colorized field.
        """
        if self.is_update_mean:
            self.theta = s_cond[:-1]
            self.estimated_mean = s_cond[-1]
        else:
            self.theta = s_cond

        z = gd_parametrize(self.W, get_gd_weights(self.theta))
        return self.cov.colorize(z) + self.estimated_mean

    def _dtransform_vec(
        self, s_raw: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat:
        """Return the transform 1st derivative times a vector as a 1-D vector."""
        return np.zeros_like(s_raw)

    def _dbacktransform_vec(
        self, s_cond: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat:
        r"""
        Return the backtransform 1st derivative times a vector.

        Uses the chain rule through ``backtransform(theta) = cov.colorize(z(theta))
        + mean``, with ``z(theta) = gd_parametrize(W, get_gd_weights(theta))``::

            dJ/dtheta = (dz/dtheta)^T @ (dcolorize/dz)^T @ gradient
                      = d_gd_parametrize_mat_vec(W, theta,
                        cov.colorize_adjoint(gradient))

        and ``dJ/d(mean) = sum(gradient)`` (since the mean term is additive
        and applies identically to every entry of the field).

        Requires ``cov.colorize_adjoint`` (added to ``covmats`` >= 0.5,
        mirroring ``matvec``/``rmatvec``; see the module docstring).

        Raises
        ------
        NotImplementedError
            If `cov` does not expose `colorize_adjoint` (e.g. a sparse
            Cholesky/precision-Cholesky backend for which the corresponding
            `SparseCholeskyFactor` adjoint is not implemented yet).
        """
        if not hasattr(self.cov, "colorize_adjoint"):
            raise NotImplementedError(
                "GDPNCS._dbacktransform_vec requires `cov.colorize_adjoint`, "
                f"which is not available on {type(self.cov).__name__}. See the "
                "module docstring notes on `GDPNCS`/`GDPCS` and `covmats`."
            )
        out = d_gd_parametrize_mat_vec(
            self.W, self.theta, self.cov.colorize_adjoint(gradient)
        )
        if self.is_update_mean:
            return np.hstack((out, np.sum(gradient)))
        return out

    def _dbacktransform_inv_vec(
        self, s_cond: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat:
        """
        Return the inverse of the backtransform 1st derivative times a vector.

        Raises
        ------
        NotImplementedError
            Always: this operation is not implemented for GDPNCS (as in the
            previous SPDE-based implementation). Contact the developers for
            details if needed.
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
            Array of shape (N_cond, 2) with very loose (near +/-inf) bounds,
            since `theta`/`estimated_mean` are not meaningfully bounded by
            the field's own bounds.
        """
        n_cond = self.theta.size
        if self.is_update_mean:
            n_cond += 1
        bounds = np.zeros((n_cond, 2))
        bounds[:, 0] = -1e100  # np.inf
        bounds[:, 1] = 1e100  # np.inf
        return bounds

    def smart_copy(self) -> GDPNCS:
        """Return a copy of the instance, deep-copying `theta` and `estimated_mean`."""
        cp = copy.copy(self)
        cp.theta = copy.deepcopy(cp.theta)
        cp.estimated_mean = copy.deepcopy(cp.estimated_mean)
        return cp


class GDPCS(GDPNCS):
    """
    Apply a Gradual Deformation parametrization for a conditional field.

    Same as :class:`GDPNCS`, but the field is conditioned to noisy point (or
    more general linear) observations, via Matheron's rule / pathwise
    conditioning: ``z_cond = z_u + cov @ H^T @ A^{-1} @ (d - H @ z_u -
    eps_u)``, where ``z_u`` is the unconditional draw, ``H`` the observation
    operator, ``A = H cov H^T + R`` the data-space system, and ``eps_u`` a
    draw of the observation noise. This generalizes the former
    SPDE/precision-matrix only implementation (``spde.simu_c``) to any
    ``covmats`` backend.

    Unlike an earlier version of this class, Matheron's rule is implemented
    directly here (drawing ``eps_u`` once at construction time and solving
    the data-space system with conjugate gradients -- the same machinery
    :meth:`_dbacktransform_vec` needs anyway) rather than by delegating to
    :func:`covmats.conditional_simulate` with a ``colorize_fn`` override and
    a reseeded RNG. The previous approach worked, but its reproducibility
    depended on the *internal, undocumented* draw order of
    ``conditional_simulate`` (it draws a throwaway `colorize_dim`-sized
    array, then ``eps_u``): reusing a fixed seed only reproduces the same
    ``eps_u`` as long as that internal order/count never changes. Drawing
    ``eps_u`` ourselves removes that coupling entirely.

    Notes
    -----
    ``eps_u`` is drawn once in :meth:`__init__` and reused on every call to
    :meth:`_backtransform`, so the conditioned field is a smooth,
    deterministic function of ``theta`` (and ``estimated_mean``) alone, as
    required for gradual-deformation optimization.

    :meth:`_dbacktransform_vec` differentiates through the full conditioning
    system (not just ``colorize``, since the correction term also depends on
    ``theta``); see that method's docstring for the derivation. It requires
    ``cov.colorize_adjoint`` for the same reason as :class:`GDPNCS`.
    """

    def __init__(
        self,
        ne: int,
        cov: covmats.CovarianceMatrix,
        obs_op: Union[LinearOperator, NDArrayFloat],
        obs_values: NDArrayFloat,
        obs_cov: Union[covmats.CovarianceMatrix, float, NDArrayFloat],
        estimated_mean: float = 0.0,
        theta: Optional[NDArrayFloat] = None,
        random_state: Optional[
            Union[int, np.random.Generator, np.random.RandomState]
        ] = None,
        is_update_mean: bool = True,
    ) -> None:
        """
        Initialize the instance.

        Parameters
        ----------
        ne : int
            Number of independent white noise realizations combined by the
            gradual deformation.
        cov : covmats.CovarianceMatrix
            Prior covariance representation of the field, with shape (n, n).
        obs_op : Union[LinearOperator, NDArrayFloat]
            Observation/measurement operator ``H``, with shape
            ``(n_obs, n)``. Use :func:`covmats.make_point_observation_operator`
            for plain point observations at grid/discretization nodes.
        obs_values : NDArrayFloat
            Observed data, with shape ``(n_obs,)``.
        obs_cov : Union[covmats.CovarianceMatrix, float, NDArrayFloat]
            Observation-error covariance ``R``. A float or 1D array of length
            `n_obs` is treated as independent per-observation variances.
        estimated_mean : float, optional
            Initial estimated (constant) prior mean of the field.
            The default is 0.0.
        theta : Optional[NDArrayFloat], optional
            Initial gradual deformation parameter, with size `ne - 1`. If
            None, it is initialized so that all Ne realizations are equally
            weighted. The default is None.
        random_state : Optional[Union[int, np.random.Generator, np.random.RandomState]]
            Pseudorandom number generator state used both to draw the
            ensemble of `ne` white noises (as in :class:`GDPNCS`) and to
            draw the fixed observation-noise realization `eps_u` used by
            Matheron's rule (see class notes). The default is None.
        is_update_mean : bool, optional
            Whether the (constant) field mean is also adjusted alongside
            `theta`. The default is True.
        """
        super().__init__(
            ne=ne,
            cov=cov,
            estimated_mean=estimated_mean,
            theta=theta,
            random_state=random_state,
            is_update_mean=is_update_mean,
        )

        self.obs_op = obs_op
        self.obs_values = obs_values
        self.obs_cov = obs_cov

        # Fixed observation-noise draw (`eps_u` in Matheron's rule): drawn
        # once here, directly, with our own rng, and reused on every call to
        # `_backtransform`/`_dbacktransform_vec`, so the conditioned field is
        # a smooth, deterministic function of `theta` (and `estimated_mean`)
        # alone. See the class docstring for why this is drawn directly
        # rather than by reseeding `covmats.conditional_simulate`'s RNG.
        self._eps_u: NDArrayFloat = self._draw_eps_u(check_random_state(random_state))

    def _draw_eps_u(
        self, rng: Union[np.random.Generator, np.random.RandomState]
    ) -> NDArrayFloat:
        """
        Draw one realization of the observation noise `eps_u` ~ N(0, R).

        Parameters
        ----------
        rng : Union[np.random.Generator, np.random.RandomState]
            Already-constructed random number generator, e.g. as returned by
            `check_random_state`.

        Returns
        -------
        NDArrayFloat
            A draw of `eps_u`, with shape `(n_obs,)`.
        """
        if isinstance(self.obs_cov, covmats.CovarianceMatrix):
            return self.obs_cov.sample_mvnormal(shape=[1], random_state=rng)[0]
        _obs_cov_arr = np.asarray(self.obs_cov, dtype=float)
        return np.sqrt(_obs_cov_arr) * rng.standard_normal(np.size(self.obs_values))

    def _get_obs_op_as_linop(self) -> LinearOperator:
        """Return `obs_op` as a `LinearOperator`, wrapping it if needed."""
        return (
            self.obs_op
            if isinstance(self.obs_op, LinearOperator)
            else sp.sparse.linalg.aslinearoperator(np.asarray(self.obs_op))
        )

    def _get_obs_cov_matvec(self) -> Callable[[NDArrayFloat], NDArrayFloat]:
        """Return a callable computing `R @ x`, whatever `obs_cov`'s type."""
        if isinstance(self.obs_cov, covmats.CovarianceMatrix):
            return self.obs_cov.matvec
        _obs_cov_arr = np.asarray(self.obs_cov, dtype=float)
        return lambda x: _obs_cov_arr * x

    def _solve_data_space_system(
        self,
        H: LinearOperator,
        r_matvec: Callable[[NDArrayFloat], NDArrayFloat],
        rhs: NDArrayFloat,
    ) -> NDArrayFloat:
        r"""
        Solve ``(H cov H^T + R) @ x = rhs`` for `x`, via conjugate gradients.

        Shared by :meth:`_backtransform` (Matheron's rule correction) and
        :meth:`_dbacktransform_vec` (its adjoint): both need a solve against
        the same data-space system, and neither ever assembles it, `cov`, or
        the posterior covariance densely.

        Parameters
        ----------
        H : LinearOperator
            Observation operator, as returned by
            :meth:`_get_obs_op_as_linop`.
        r_matvec : Callable[[NDArrayFloat], NDArrayFloat]
            Callable computing ``R @ x``, as returned by
            :meth:`_get_obs_cov_matvec`.
        rhs : NDArrayFloat
            Right-hand side, with shape `(n_obs,)`.

        Returns
        -------
        NDArrayFloat
            The solution `x`, with shape `(n_obs,)`.

        Raises
        ------
        RuntimeError
            If the conjugate-gradient solve does not converge.
        """
        n_obs = H.shape[0]

        def _a_matvec(x: NDArrayFloat) -> NDArrayFloat:
            return H.matvec(self.cov.matvec(H.rmatvec(x))) + r_matvec(x)

        a_linop = LinearOperator((n_obs, n_obs), matvec=_a_matvec, dtype=float)
        sol, info = sp.sparse.linalg.cg(a_linop, rhs, rtol=1e-10)
        if info != 0:
            raise RuntimeError(
                "GDPCS: the conjugate-gradient solve for the data-space "
                f"system did not converge (info={info})."
            )
        return sol

    def _backtransform(self, s_cond: NDArrayFloat) -> NDArrayFloat:
        """
        Apply the back-preconditioning/parametrization.

        Implements Matheron's rule directly: ``z_cond = z_u + cov @ H^T @
        A^{-1} @ (d - H @ z_u - eps_u)``, with ``z_u`` the GD-parametrized
        unconditional draw and ``eps_u`` the fixed observation-noise draw
        from :meth:`__init__` (see the class docstring).

        Parameters
        ----------
        s_cond : NDArrayFloat
            Conditioned (transformed) parameter values: the GD parameter
            `theta` (plus `estimated_mean` if `is_update_mean` is True).

        Returns
        -------
        NDArrayFloat
            Non-conditioned (transformed) parameter values, i.e. the
            GD-parametrized field conditioned to the observation data.
        """
        if self.is_update_mean:
            self.theta = s_cond[:-1]
            self.estimated_mean = s_cond[-1]
        else:
            self.theta = s_cond

        z = gd_parametrize(self.W, get_gd_weights(self.theta))
        z_u = self.cov.colorize(z) + self.estimated_mean

        H = self._get_obs_op_as_linop()
        r_matvec = self._get_obs_cov_matvec()
        resid = self.obs_values - (H.matvec(z_u) + self._eps_u)
        lam = self._solve_data_space_system(H, r_matvec, resid)
        return z_u + self.cov.matvec(H.rmatvec(lam))

    def _dbacktransform_vec(
        self, s_cond: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat:
        r"""
        Return the backtransform 1st derivative times a vector.

        Unlike :meth:`GDPNCS._dbacktransform_vec`, the conditioning
        correction term of Matheron's rule itself depends on `theta` through
        the unconditional draw, so the chain rule must go through the whole
        conditioning system, not just through ``cov.colorize``.

        Writing ``A = H cov H^T + R`` (the ``(n_obs, n_obs)`` system solved
        by :meth:`_solve_data_space_system`) and
        ``z_cond(z) = mean + (I - cov H^T A^{-1} H) colorize(z) + const``
        (the part of :meth:`_backtransform` depending on the GD white noise
        `z`, with everything not depending on `z` -- the data, `eps_u` --
        folded into the constant), the adjoint is::

            lam = A^{-1} @ (H @ (cov @ gradient))
            corrected = gradient - H^T @ lam
            d(theta) = d_gd_parametrize_mat_vec(
                W, theta, cov.colorize_adjoint(corrected)
            )
            d(mean) = sum(corrected)

        This was validated against finite differences of
        :meth:`_backtransform`.

        Raises
        ------
        NotImplementedError
            If `cov` does not expose `colorize_adjoint` (see
            :meth:`GDPNCS._dbacktransform_vec`).
        RuntimeError
            If the conjugate-gradient solve for `lam` does not converge.
        """
        if not hasattr(self.cov, "colorize_adjoint"):
            raise NotImplementedError(
                "GDPCS._dbacktransform_vec requires `cov.colorize_adjoint`, "
                f"which is not available on {type(self.cov).__name__}. See the "
                "module docstring notes on `GDPNCS`/`GDPCS` and `covmats`."
            )
        H = self._get_obs_op_as_linop()
        r_matvec = self._get_obs_cov_matvec()
        lam = self._solve_data_space_system(
            H, r_matvec, H.matvec(self.cov.matvec(gradient))
        )
        corrected = gradient - H.rmatvec(lam)
        out = d_gd_parametrize_mat_vec(
            self.W, self.theta, self.cov.colorize_adjoint(corrected)
        )
        if self.is_update_mean:
            return np.hstack((out, np.sum(corrected)))
        return out

    def _dbacktransform_inv_vec(
        self, s_cond: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat:
        """
        Return the inverse of the backtransform 1st derivative times a vector.

        Raises
        ------
        NotImplementedError
            Always: this operation is not implemented for GDPCS (as in the
            previous SPDE-based implementation). Contact the developers for
            details if needed.
        """
        raise NotImplementedError(
            "_dbacktransform_inv_vec is not implemented for "
            "GDPCS! Contact developers for detail!"
        )


class SubSelector(Preconditioner):
    """Apply a selection on the input field, keeping the rest of it fixed."""

    def __init__(self, node_numbers: NDArrayInt, grid: RectilinearGrid) -> None:
        """
        Initialize the instance.

        Parameters
        ----------
        node_numbers : NDArrayInt
            Node(s) to sample/select from the field.
        grid : RectilinearGrid
            Grid defining the size of the field to be sampled.
        """
        self.node_numbers = np.array(node_numbers)
        self.field_size: int = grid.n_grid_cells
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
        """Return the transform 1st derivative times a vector as a 1-D vector."""
        assert np.size(s_raw) == self.field_size
        assert np.size(gradient) == self.node_numbers.size
        out = np.zeros(self.field_size)
        out[self.node_numbers] = gradient
        return out

    def _dbacktransform_vec(
        self, s_cond: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat:
        """Return the backtransform 1st derivative times a vector."""
        assert gradient.size == self.field_size
        return gradient[self.node_numbers]

    def _dbacktransform_inv_vec(
        self, s_cond: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat:
        """Return the inverse of the backtransform 1st derivative times a vector."""
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

        This is a development tool. The `dbacktransform_inv_vec` check (4) is
        skipped, since sub-selection is not invertible (values outside of
        `node_numbers` are lost, not restored, by `transform`/`backtransform`).

        Parameters
        ----------
        lbounds : Union[float, NDArrayFloat]
            Lower bound(s) used to generate random test values.
        ubounds : Union[float, NDArrayFloat]
            Upper bound(s) used to generate random test values.
        shape : Optional[Union[int, Sequence[int]]], optional
            Shape of the generated test values. The default is None.
        rtol : float, optional
            Relative tolerance used for all correctness checks.
            The default is 1e-5.
        eps : Optional[float], optional
            The epsilon for the computation of the approximated preconditioner first
            derivative by finite difference. The default is None.

        Raises
        ------
        ValueError
            If one of the backconditioner of the gradient conditioner are incorrect.
        """
        super()._test_preconditioner(lbounds, ubounds, shape, rtol, eps, [4])


class Slicer(SubSelector):
    """Apply a slicing to the field of values, selecting a rectilinear sub-domain."""

    def __init__(
        self,
        grid: RectilinearGrid,
        span: Union[NDArrayInt, Tuple[slice, slice], NDArrayBool] = (
            slice(None),
            slice(None),
        ),
    ) -> None:
        """
        Initialize the instance.

        Parameters
        ----------
        grid : RectilinearGrid
            Grid defining the size and shape of the field to be sliced.
        span : Union[NDArrayInt, Tuple[slice, slice], NDArrayBool], optional
            Slice, boolean mask, or index array applied to the (Fortran
            order) reshaped field to select the nodes to keep. The default
            selects the whole field (``(slice(None), slice(None))``).
        """
        field_size = grid.n_grid_cells
        node_numbers = np.arange(field_size).reshape(grid.shape, order="F")[span]
        super().__init__(node_numbers, grid)


def gaussian_cfd(x: NDArrayFloat, mu: float, std: float) -> NDArrayFloat:
    """
    Return the cumulative distribution function (CDF) of a Gaussian distribution.

    Parameters
    ----------
    x : NDArrayFloat
        Input values at which to evaluate the CDF.
    mu : float
        Mean of the Gaussian distribution.
    std : float
        Standard deviation of the Gaussian distribution.

    Returns
    -------
    NDArrayFloat
        CDF values, in ``[0, 1]``.
    """
    return 0.5 * (1.0 + sp.special.erf((x - mu) / (std * np.sqrt(2))))


def gaussian_cfd_inv(x: NDArrayFloat, mu: float, std: float) -> NDArrayFloat:
    """
    Return the inverse cumulative distribution function (quantile function).

    Parameters
    ----------
    x : NDArrayFloat
        Input probabilities, in ``[0, 1]``.
    mu : float
        Mean of the Gaussian distribution.
    std : float
        Standard deviation of the Gaussian distribution.

    Returns
    -------
    NDArrayFloat
        Values of the Gaussian quantile function (inverse of :func:`gaussian_cfd`).
    """
    return sp.special.erfinv(2.0 * x - 1.0) * (std * np.sqrt(2)) + mu


def gaussian_cfd_inv_deriv(x: NDArrayFloat, mu: float, std: float) -> NDArrayFloat:
    """
    Return the derivative (w.r.t. x) of :func:`gaussian_cfd_inv`.

    Parameters
    ----------
    x : NDArrayFloat
        Input probabilities, in ``[0, 1]``.
    mu : float
        Mean of the Gaussian distribution. Unused (the derivative does not
        depend on the mean), kept for signature symmetry with
        :func:`gaussian_cfd_inv`.
    std : float
        Standard deviation of the Gaussian distribution.

    Returns
    -------
    NDArrayFloat
        Derivative of the Gaussian quantile function w.r.t. `x`.
    """
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
        """Return the transform 1st derivative times a vector as a 1-D vector."""
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
        """Return the backtransform 1st derivative times a vector."""
        return (
            (self.ud_ubound - self.ud_lbound)
            * np.exp(-((s_cond - self.gd_mu) ** 2) / (2 * self.gd_std**2))
            / (np.sqrt(2 * np.pi) * self.gd_std)
        ) * gradient

    def _dbacktransform_inv_vec(
        self, s_cond: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat:
        """Return the inverse of the backtransform 1st derivative times a vector."""
        return gradient / (
            (self.ud_ubound - self.ud_lbound)
            * np.exp(-((s_cond - self.gd_mu) ** 2) / (2 * self.gd_std**2))
            / (np.sqrt(2 * np.pi) * self.gd_std)
        )


class BoundsClipper(Preconditioner):
    """Clip the parameter to the given bounds (non-invertible transform)."""

    LBOUND_RAW: float = -np.inf
    UBOUND_RAW: float = +np.inf

    def __init__(
        self,
        lbounds: Union[float, NDArrayFloat],
        ubounds: Union[float, NDArrayFloat],
    ) -> None:
        """
        Initialize the instance.

        Parameters
        ----------
        lbounds : Union[float, NDArrayFloat]
            Lower bound(s) used to clip the back-transformed values.
        ubounds : Union[float, NDArrayFloat]
            Upper bound(s) used to clip the back-transformed values.
        """
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

        Raises
        ------
        ValueError
            If any value in `s_raw` is out of the given bounds (clipping
            only ever happens on `backtransform`, so `transform` should
            never see out-of-bounds values under normal use).
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
            Non-conditioned (transformed) parameter values, clipped to
            ``[lbounds, ubounds]``.
        """
        return s_cond.clip(self.lbounds, self.ubounds)

    def _dtransform_vec(
        self, s_raw: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat:
        """Return the transform 1st derivative times a vector as a 1-D vector."""
        return gradient

    def _dbacktransform_vec(
        self, s_cond: NDArrayFloat, gradient: NDArrayFloat
    ) -> NDArrayFloat:
        """Return the backtransform 1st derivative times a vector."""
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
        """Return the inverse of the backtransform 1st derivative times a vector."""
        # should have the same effect as gradient / np.sign(s_cond)
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
            Array of shape (N_s, 2), unchanged (clipping does not affect the
            bounds themselves).
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
        Test if the backconditioner derivative is correct.

        This is a development tool. Only the finite-difference check on
        `dbacktransform_vec` is performed: `backtransform` is not the
        inverse of `transform` here (clipping is not invertible), so the
        other checks from :meth:`Preconditioner._test_preconditioner` do
        not apply.

        Parameters
        ----------
        lbounds : Union[float, NDArrayFloat]
            Lower bound(s) used to generate random test values.
        ubounds : Union[float, NDArrayFloat]
            Upper bound(s) used to generate random test values.
        shape : Optional[Union[int, Sequence[int]]], optional
            Shape of the generated test values. The default is None.
        rtol : float, optional
            Relative tolerance used for the correctness check.
            The default is 1e-5.
        eps : Optional[float], optional
            The epsilon for the computation of the approximated preconditioner first
            derivative by finite difference. The default is None.

        Raises
        ------
        AssertionError
            If the backconditioner derivative does not match its finite
            difference approximation.
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
    """
    Scale the given preconditioner with the scaling factor.

    Parameters
    ----------
    scaling_factor : float
        Multiplicative scaling factor applied after `pcd`.
    pcd : Preconditioner
        Preconditioner to scale. It is deep-copied (via
        :meth:`Preconditioner.smart_copy`) so the original instance is not
        mutated.

    Returns
    -------
    Preconditioner
        A :class:`ChainedTransforms` applying `pcd` then a
        :class:`LinearTransform` of slope `scaling_factor` and zero intercept.
    """
    return ChainedTransforms(
        [pcd.smart_copy(), LinearTransform(slope=scaling_factor, y_intercept=0.0)]
    )


@dataclass
class GradientScalerConfig:
    """

    Attributes
    ----------
    max_change_target : float
        Maximum update desired on the parameter values.
    pcd_change_eval: Preconditioner
        Preconditioner to apply to evaluate the change. This is typically useful
        when the target is defined on a logscale or after a linear scaling.
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

    max_change_target: float
    pcd_change_eval: Preconditioner = field(default_factory=NoTransform)
    max_workers: int = 50
    rtol: float = 0.05
    lb: float = 1e-10
    ub: float = 1e10
    n_samples_in_first_round: int = 50


def get_max_update(
    scaling_factor: float,
    pcd: Preconditioner,
    s_nc: NDArrayFloat,
    grad_nc: NDArrayFloat,
    gsc: Optional[GradientScalerConfig] = None,
    lb_nc: Optional[Union[NDArrayFloat, float]] = None,
    ub_nc: Optional[Union[NDArrayFloat, float]] = None,
) -> float:
    """
    Get the max update of parameter values with gradient descent on conditioned values.

    Parameters
    ----------
    scaling_factor: float
        Scaling factor for the parameter.
    pcd : Preconditioner
        Preconditioner.
    s_nc : NDArrayFloat
        Non conditioned parameter values.
    grad_nc : NDArrayFloat
        Gradient of the objective function with respect to the non conditioned parameter
        values.
    gsc: Optional[GradientScalerConfig]
        Configuration for the gradient scaling. The default is None.
    lb_nc: Optional[NDArrayFloat]
        Lower bound on the non conditioned parameter. The default is None.
    ub_nc: Optional[NDArrayFloat]
        Upper bound on the non conditioned parameter. The default is None.
    Returns
    -------
    float
        Maximum update of the parameter values.
    """
    pcd = pcd.smart_copy()
    pcd_scaled = scale_pcd(scaling_factor, pcd)
    s_cond = pcd_scaled(s_nc)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        s1 = pcd_scaled.backtransform(
            s_cond - pcd_scaled.dbacktransform_vec(s_cond, grad_nc)
        )
    if lb_nc is not None or ub_nc is not None:
        s1 = np.clip(
            s1,
            a_min=lb_nc,
            a_max=ub_nc,
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if gsc is not None:
            diff = gsc.pcd_change_eval(s1) - gsc.pcd_change_eval(s_nc)
        else:
            diff = s1 - s_nc
        return float(sp.linalg.norm(diff, ord=np.inf))


def cost_fun(
    scaling_factor: float,
    pcd: Preconditioner,
    s_nc: NDArrayFloat,
    grad_nc: NDArrayFloat,
    max_change_target,
    gsc: Optional[GradientScalerConfig] = None,
    lb_nc: Optional[Union[NDArrayFloat, float]] = None,
    ub_nc: Optional[Union[NDArrayFloat, float]] = None,
) -> float:
    """
    Cost function used to search for the scaling factor enforcing a target update.

    Returns ``log((get_max_update(...) - max_change_target) ** 2 + 1)``: a
    smooth, always-positive function of `scaling_factor` that is minimized
    when :func:`get_max_update` matches `max_change_target`.

    Parameters
    ----------
    scaling_factor : float
        Scaling factor for the parameter, forwarded to :func:`get_max_update`.
    pcd : Preconditioner
        Preconditioner, forwarded to :func:`get_max_update`.
    s_nc : NDArrayFloat
        Non conditioned parameter values, forwarded to :func:`get_max_update`.
    grad_nc : NDArrayFloat
        Gradient of the objective function with respect to the non
        conditioned parameter values, forwarded to :func:`get_max_update`.
    max_change_target : float
        Target maximum update of the parameter values.
    gsc : Optional[GradientScalerConfig], optional
        Configuration for the gradient scaling, forwarded to
        :func:`get_max_update`. The default is None.
    lb_nc : Optional[Union[NDArrayFloat, float]], optional
        Lower bound on the non conditioned parameter, forwarded to
        :func:`get_max_update`. The default is None.
    ub_nc : Optional[Union[NDArrayFloat, float]], optional
        Upper bound on the non conditioned parameter, forwarded to
        :func:`get_max_update`. The default is None.

    Returns
    -------
    float
        The (always positive, log-scaled) cost value.
    """
    return np.log(
        (
            get_max_update(
                scaling_factor,
                pcd,
                s_nc=s_nc,
                grad_nc=grad_nc,
                gsc=gsc,
                lb_nc=lb_nc,
                ub_nc=ub_nc,
            )
            - max_change_target
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


def is_picklable(obj) -> bool:
    """
    Return whether `obj` can be pickled.

    Used to decide whether :func:`get_factor_enforcing_grad_inf_norm` can
    parallelize its search over scaling factors using a
    :class:`~concurrent.futures.ProcessPoolExecutor` (which requires
    pickling `obj`), or must fall back to sequential evaluation.

    Parameters
    ----------
    obj : Any
        Object whose picklability is tested.

    Returns
    -------
    bool
        True if `obj` can be pickled, False otherwise.
    """
    try:
        pickle.dumps(obj)

    except (pickle.PicklingError, TypeError):
        return False
    return True


def get_factor_enforcing_grad_inf_norm(
    s_nc: NDArrayFloat,
    grad_nc: NDArrayFloat,
    pcd: Preconditioner,
    gsc: GradientScalerConfig,
    lb_nc: Optional[Union[NDArrayFloat, float]] = None,
    ub_nc: Optional[Union[NDArrayFloat, float]] = None,
    logger: Optional[logging.Logger] = None,
) -> float:
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
    lb_nc: Optional[NDArrayFloat]
        Lower bound on the non conditioned parameter. The default is None.
    ub_nc: Optional[NDArrayFloat]
        Upper bound on the non conditioned parameter. The default is None.
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
            1.0, pcd, s_nc, grad_nc, gsc, lb_nc=lb_nc, ub_nc=ub_nc
        )
        logger.info("Initial scaling factor = 1.0")
        logger.info(f"Initial maximum change   = {init_max_change:.2e}")
        logger.info(f"Objective maximum change = {gsc.max_change_target:.2e}\n")

    # If the initial maximum change already respects the objective, then leave
    if (
        np.abs(
            get_relative_error(
                init_max_change,
                gsc.max_change_target,
            )
        )
        <= gsc.rtol
    ):
        if logger is not None:
            logger.info("Target already fulfilled, skipping optimization \n")
        return 1.0

    # step 1: explore
    scaling_factor = 1.0  # initial guess
    round = 1

    def get_pcd() -> Generator:
        while True:
            yield pcd.smart_copy()

    def get_s_nc() -> Generator:
        while True:
            yield s_nc

    def get_grad_nc() -> Generator:
        while True:
            yield grad_nc

    def get_gsc() -> Generator:
        while True:
            yield gsc

    def get_lb_nc() -> Generator:
        while True:
            yield lb_nc

    def get_ub_nc() -> Generator:
        while True:
            yield ub_nc

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

    lb = copy.deepcopy(gsc.lb)
    ub = copy.deepcopy(gsc.ub)

    while (
        np.abs(
            get_relative_error(
                get_max_update(
                    scaling_factor, pcd, s_nc, grad_nc, gsc, lb_nc=lb_nc, ub_nc=ub_nc
                ),
                gsc.max_change_target,
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
            return 1.0

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
                        _scaling_factor,
                        pcd,
                        s_nc,
                        grad_nc,
                        gsc,
                        lb_nc=lb_nc,
                        ub_nc=ub_nc,
                    )
                )
        else:
            with ProcessPoolExecutor(max_workers=_max_workers) as executor:
                max_s_nc_updates = list(
                    executor.map(
                        get_max_update,
                        scaling_factors,
                        get_pcd(),
                        get_s_nc(),
                        get_grad_nc(),
                        get_gsc(),
                        get_lb_nc(),
                        get_ub_nc(),
                    )
                )

        squared_diff = (np.array(max_s_nc_updates) - gsc.max_change_target) ** 2
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
            _re = get_relative_error(max_s_nc_updates[argmin], gsc.max_change_target)
            logger.info(f"Post round {round}: Rel. error to target = {_re:.2%}\n")

        # update the round number
        round += 1

    return scaling_factor
