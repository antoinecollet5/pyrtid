from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Colormap

from pyrtid.utils import NDArrayFloat


def plot_percentiles(
    ax: Axes,
    data: NDArrayFloat,
    x: Optional[NDArrayFloat] = None,
    cmap: Colormap = plt.get_cmap("Reds"),
    method="median_unbiased",
) -> Axes:
    """
    Plot the percentiles of a timeseries using color gradient.

    Parameters
    ----------
    ax : Axes
        Axis on which to plot.
    cmap : Colormap
        Colormap to use.
    x : NDArrayFloat
        Number of points (nx) in each timeseries.
    data : NDArrayFloat
        Array of timeseries with shape (nx, n_sample), n_sample being the number of
        timeseries in the sample.

    Returns
    -------
    Axes
        Updated input axis.
    """

    n = 19  # 9 bins + the P50 that we won't use
    percentiles = np.linspace(start=5.0, stop=95.0, num=n)
    s_dist = np.percentile(data, q=percentiles, method=method, axis=1)

    if x is None:
        _x: NDArrayFloat = np.arange(data.shape[0])
    else:
        _x = x

    # plot the color ranges
    for i in range(int(n / 2)):
        ax.fill_between(
            _x,
            s_dist[i],
            s_dist[-(i + 1)],
            color=cmap(0.1 + i / n * 2 / 1.6),
            label=f"Pct{percentiles[i * 2 + 1]}",
        )

    # plot the median
    ax.plot(_x, s_dist[int(n / 2)], linestyle="-", c="k", label="Median")

    return ax
