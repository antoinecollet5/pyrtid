import matplotlib.pyplot as plt
import numpy as np

from pyrtid.plot import plot_observed_vs_simulated


def test_plot_observed_s_simulated() -> None:
    obs = np.random.normal(loc=1, scale=1, size=200)
    pred_after_inversion = obs + np.random.normal(loc=0, scale=0.1, size=200)
    pred_before_inversion = obs + np.random.normal(loc=0, scale=0.5, size=200)

    # Test with minimum arguments
    fig, ax = plt.subplots()
    plot_observed_vs_simulated(ax, obs, pred_after_inversion)

    # Test with all arguments
    fig, ax = plt.subplots()
    plot_observed_vs_simulated(
        ax, obs, pred_after_inversion, pred_before_inversion, units="my unit"
    )
    ax.legend()
