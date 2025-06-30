"""Test the priors implementation."""

import re
from contextlib import nullcontext as does_not_raise
from typing import Optional

import numpy as np
import pytest
from pyrtid.regularization import (
    ConstantPriorTerm,
    DriftMatrix,
    EnsembleMeanPriorTerm,
    LinearDriftMatrix,
    MeanPriorTerm,
    NullPriorTerm,
)
from pyrtid.utils.grid import RectilinearGrid


def test_null_prior() -> None:
    prior = NullPriorTerm()
    assert prior.get_values(np.ones(45)) == 0.0
    np.testing.assert_equal(prior.get_gradient_dot_product(np.ones(45)), np.zeros(45))


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

    np.testing.assert_equal(prior.get_gradient_dot_product(np.ones(45)), np.zeros(45))


def test_mean_prior() -> None:
    prior = MeanPriorTerm()

    np.testing.assert_allclose(prior.get_values(np.ones(45) * 9.6), np.ones(45) * 9.6)
    np.testing.assert_allclose(prior.get_values(np.ones(22) * 8.3), np.ones(22) * 8.3)

    np.testing.assert_allclose(prior.get_gradient_dot_product(np.ones(45)), np.ones(45))


@pytest.mark.parametrize(
    "shape,ensemble,expected_values,vector,expected_exception",
    [
        (
            (10,),
            np.array([]),
            0.0,
            np.array([]),
            pytest.raises(
                ValueError,
                match=re.escape(
                    "The shape of an EnsembleMeanPriorTerm should be (N_s, N_e)"
                    " with N_s the number of adjuted values and N_e the number of"
                    " members in the ensemble."
                ),
            ),
        ),
        (
            (10, 20),
            np.ones((2, 20)),
            0.0,
            np.array([]),
            pytest.raises(
                ValueError, match=re.escape(r"Expected shape (10, 20), got (2, 20).")
            ),
        ),
        (
            (10, 20),
            np.ones((10, 20)) * 2.0,
            np.ones((10, 1)) * 2.0,
            np.ones((10)),
            does_not_raise(),
        ),
        (
            (10, 20),
            np.ones((10, 20)) * 2.0,
            np.ones((10, 1)) * 2.0,
            np.ones(5),
            pytest.raises(
                ValueError, match=re.escape("Expected a vector of size 10, got (5,).")
            ),
        ),
    ],
)
def test_ensemble_mean_prior(
    shape, ensemble, expected_values, vector, expected_exception
) -> None:
    with expected_exception:
        prior = EnsembleMeanPriorTerm(shape)
        prior.shape == shape
        np.testing.assert_array_equal(prior.get_values(ensemble), expected_values)
        assert prior.get_gradient_dot_product(vector).shape == vector.shape


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


def test_linear_drift_matrix2() -> None:
    nx = 100
    ny = 100
    grid = RectilinearGrid(
        x0=0.0, y0=0.0, z0=0.0, dx=5.0, dy=5.0, dz=1.0, nx=nx, ny=ny, nz=1
    )
    # coordinates 2d
    coords = grid.center_coords_2d.reshape(2, -1, order="F")

    # Linear trend: b0 + b1 * x + b2 * y
    trend = DriftMatrix(
        np.concatenate([np.ones((1, nx * ny)), coords]).T,
        beta=np.array([219, 0.1, -0.1]),
    )
    trend.get_values(np.ones(nx * ny))

    assert trend.s_dim == nx * ny
