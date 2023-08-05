# -*- coding: utf-8 -*-
import numdifftools as nd
import numpy as np
import pytest

from pyrtid.utils import finite_gradient
from pyrtid.utils.finite_differences import rosen, rosen_gradient, rosen_hessian


@pytest.mark.parametrize("values", [([1.0, 1.0]), (10.0, 10.0), (0.0, 79.9)])
def test_gradient(values) -> None:
    np.testing.assert_allclose(
        rosen_gradient(np.array(values)),
        finite_gradient(np.array(values), rosen),
        atol=1e-4,
    )


@pytest.mark.parametrize("values", [([1.0, 1.0]), ([1.5, -0.5]), ([0.0, 1.9])])
def test_hessian(values) -> None:
    np.testing.assert_allclose(
        rosen_hessian(np.array(values)), nd.Hessian(rosen)(values), atol=1e-4
    )
