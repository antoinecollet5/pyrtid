import numpy as np

from pyrtid.inverse.preconditioner import expit, logit
from pyrtid.utils import NDArrayFloat


def test_logit() -> NDArrayFloat:
    # Example 1
    min1 = 0.51
    max1 = 5049
    x = np.linspace(min1, max1, 50)
    y = logit(x, min1 - 0.1 * min1, max1 + 0.1 * min1)

    # Example 2
    min2 = 1e-9
    max2 = 1e-4

    x = np.linspace(min2, max2, 50)
    y = logit(x, min2 - 0.1 * min2, max2 + 0.1 * min2)
    y2 = expit(x, min2 - 0.1 * min2, max2 + 0.1 * min2)

    return y + y2
