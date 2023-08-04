"""Test the drift matrices implementation."""
from typing import Optional

import numpy as np
import pytest

from pyrtid.inverse.regularization import (
    ConstantPriorTerm,
    DriftMatrix,
    LinearDriftMatrix,
    MeanPriorTerm,
    NullPriorTerm,
)


def test_null_prior() -> None:
    prior = NullPriorTerm()
    assert prior.get_values(np.ones(45)) == 0.0
    assert prior.get_gradient_dot_product(np.ones(45)) == 0.0


def test_constant_prior() -> None:
    prior_values = np.ones(22) * 5.0
    prior = ConstantPriorTerm(prior_values)

    np.testing.assert_array_equal(prior.get_values(np.ones(22) * 8.3), prior_values)

    with pytest.raises(
        ValueError,
        match=(
            r"The given values have shape \(45,\) while the "
            r"constant prior has been defined with shape \(22,\)!"
        ),
    ):
        np.testing.assert_array_equal(prior.get_values(np.ones(45) * 9.6), prior_values)

    assert prior.get_gradient_dot_product(np.ones(45)) == 0.0


def test_mean_prior() -> None:
    prior = MeanPriorTerm()

    np.testing.assert_allclose(prior.get_values(np.ones(45) * 9.6), np.ones(45) * 9.6)
    np.testing.assert_allclose(prior.get_values(np.ones(22) * 8.3), np.ones(22) * 8.3)

    np.testing.assert_allclose(prior.get_gradient_dot_product(np.ones(45)), np.ones(45))


def test_drift_matrix() -> None:
    dmat = DriftMatrix(
        np.array([[2.0, 3.0, 2.0], [2.0, 3.0, 2.0]]), beta=np.array([2.0, 2.0, 2.0])
    )
    assert dmat.mat.shape == (2, 3)

    np.testing.assert_allclose(dmat.get_values(np.ones(2)), np.ones(2) * 14.0)

    with pytest.raises(
        ValueError,
        match=(
            r"The given values have size 3 while the X matrix "
            r"has been defined with shape \(2, 3\)!"
        ),
    ):
        np.testing.assert_allclose(dmat.get_values(np.ones(3)), np.ones(3))

    assert dmat.get_gradient_dot_product(np.ones(45)) == 0.0


def test_drift_matrix_no_beta() -> None:
    dmat = DriftMatrix(np.array([[2.0, 3.0, 2.0], [2.0, 3.0, 2.0]]))
    with pytest.raises(ValueError, match=r"beta is None! A value must be given."):
        np.testing.assert_allclose(dmat.get_values(np.ones(2)), np.ones(2) * 14.0)


def test_drift_matrix_wrong_beta() -> Optional[DriftMatrix]:
    with pytest.raises(
        ValueError,
        match=(
            r"beta has shape \(1,\) while it should be "
            r"shape \(3,\) to match the given coefficient matrix."
        ),
    ):
        return DriftMatrix(np.array([[2.0, 3.0, 2.0], [2.0, 3.0, 2.0]]), beta=2.0)

    with pytest.raises(
        ValueError,
        match=(
            r"beta has shape \(2,\) while it should be shape \(3,\) "
            r"to match the given coefficient matrix."
        ),
    ):
        return DriftMatrix(
            np.array([[2.0, 3.0, 2.0], [2.0, 3.0, 2.0]]), beta=np.array([2.0, 2.0])
        )


def test_linear_drift_matrix() -> None:
    pts = np.array([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0], [4.0, 1.0], [5.0, 1.0]])
    dmat = LinearDriftMatrix(pts)
    assert dmat.mat.shape == (5, 3)

    pts = np.array([[1.0, 1.0, 5.0], [2.0, 1.0, 5.0]])
    dmat = LinearDriftMatrix(pts)
    assert dmat.mat.shape == (2, 4)

    assert dmat.get_gradient_dot_product(np.ones(45)) == 0.0
