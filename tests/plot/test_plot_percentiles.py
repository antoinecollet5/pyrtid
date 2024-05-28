import matplotlib.pyplot as plt
import numpy as np
import pytest
from pyrtid.plot import plot_percentiles


@pytest.mark.parametrize("x", (None, np.arange(365) * 10))
def test_plot_percentiles(x) -> None:
    # create samples
    sample_data = (375 - 367) * np.random.random_sample((365, 50)) + 367
    # create a figure
    fig, (ax1) = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(8, 4))
    plot_percentiles(ax=ax1, data=sample_data, x=x)
