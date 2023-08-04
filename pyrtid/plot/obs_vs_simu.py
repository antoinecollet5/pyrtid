from typing import Optional

import numpy as np
from matplotlib.axes import Axes

from pyrtid.utils.types import NDArrayFloat


def plot_observed_vs_simulated(
    ax: Axes,
    obs_vector: NDArrayFloat,
    pred_vector: Optional[NDArrayFloat] = None,
    pred_vector_initial: Optional[NDArrayFloat] = None,
    units: Optional[str] = None,
) -> None:
    """
    Plot observed vs simulated data.

    Parameters
    ----------
    ax: Axes
        The ax on which to plot the data.
    obs_vector : NDArrayFloat
        The vector of observed data.
    pred_vector : Optional[NDArrayFloat], optional
        The vector of predicted data. The default is None.
    pred_vector_initial : Optional[NDArrayFloat], optional
        An optional additional vector of 'initial' predicted data.
        It allows to see the difference between before and after an inversion.
        The default is None.
    units : Optional[str], optional
        The unit of the data (for display). The default is None.

    """
    ax.set_title("obs. vs simul.", fontweight="bold")
    if pred_vector_initial is None and pred_vector is None:
        raise ValueError(
            'At least one for "pred_vector_initial" or "pred_vector" should be given !'
        )
    VERY_LARGE_NB = 1.0e40
    minobs = VERY_LARGE_NB
    maxobs = -VERY_LARGE_NB
    if pred_vector_initial is not None:
        ax.plot(obs_vector, pred_vector_initial, ".", c="r", zorder=1, label="initial")
        minobs: float = min(
            minobs, float(np.min(np.vstack((obs_vector, pred_vector_initial.ravel()))))
        )
        maxobs: float = max(
            maxobs, float(np.max(np.vstack((obs_vector, pred_vector_initial.ravel()))))
        )
    if pred_vector is not None:
        ax.plot(obs_vector, pred_vector, ".", c="b", zorder=2, label="current")
        minobs: float = min(
            minobs, float(np.min(np.vstack((obs_vector, pred_vector.ravel()))))
        )
        maxobs: float = max(
            maxobs, float(np.max(np.vstack((obs_vector, pred_vector.ravel()))))
        )

    ax.set_aspect("equal", adjustable="box")
    # ax.axis('square')
    _suffix = ""
    if units is not None:
        _suffix = f" [{units}]"
    ax.set_xlabel("observation" + _suffix, fontweight="bold")
    ax.set_ylabel("simulation" + _suffix, fontweight="bold")
    margin: float = 0.05 * np.abs(
        maxobs - minobs
    )  # 5% on each side for a nicer display
    ax.set_xlim((minobs - margin, maxobs + margin))
    ax.set_ylim(*ax.get_xlim())

    ax.plot(
        np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 20),
        np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 20),
        "k-",
    )
