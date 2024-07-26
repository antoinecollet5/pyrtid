"""Provide plot utilities for gradient comparison"""

from pathlib import Path
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import nested_grid_plotter as ngp
import numpy as np

from pyrtid.forward import Geometry
from pyrtid.utils import NDArrayFloat


def plot_2d_grad_res_adj_vs_fd(
    adj_grad: NDArrayFloat,
    fd_grad: NDArrayFloat,
    geom: Geometry,
    fname: str,
    fig_save_path: Path,
    grid_scaling: float = 1.0,
    res_scaling: Optional[float] = None,
    prod_locations: Optional[Union[NDArrayFloat, List[NDArrayFloat]]] = None,
    inj_locations: Optional[Union[NDArrayFloat, List[NDArrayFloat]]] = None,
) -> None:
    plotter = ngp.NestedGridPlotter(
        plt.figure(constrained_layout=True, figsize=(10, 8)),
        ngp.SubplotsMosaicBuilder(
            mosaic=[["ax1-1"], ["ax1-2"], ["ax1-3"]],
            sharey=True,
            sharex=True,
        ),
    )

    # We multiply the residuals so that the high residulas is just below the max values
    residuals = adj_grad - fd_grad

    if res_scaling is None:
        res_factor = 1.0
        while np.max(np.abs(adj_grad)) > np.max(np.abs(residuals)) * res_factor:
            res_factor *= 2.0
        # Make sure it is below
        res_factor /= 2.0
    else:
        res_factor = res_scaling

    ngp.multi_imshow(
        axes=plotter.axes,
        fig=plotter.fig,
        data={
            "Finite differences": fd_grad,
            "Adjoint state": adj_grad,
            f"Residuals (x {res_factor:.0e})": residuals * res_factor,
        },
        imshow_kwargs={
            "extent": [0.0, geom.nx * geom.dx, 0.0, geom.ny * geom.dy],
            "aspect": "equal",
        },
        xlabel="X [m]",
        ylabel="Z [m]",
        is_symmetric_cbar=True,
    )

    for ax in plotter.ax_dict.values():
        # Add some vertical lines to indicate the well
        if prod_locations is not None:
            for well_pos in prod_locations:
                ax.plot(
                    well_pos[0] * grid_scaling * geom.dx + geom.dx / 2,
                    well_pos[1] * grid_scaling * geom.dy + geom.dy / 2,
                    label="prod wells",
                    marker="^",
                    markersize=10,
                    c="black",
                    linestyle="none",
                )

        if inj_locations is not None:
            for well_pos in inj_locations:
                ax.plot(
                    well_pos[0] * grid_scaling * geom.dx + geom.dx / 2,
                    well_pos[1] * grid_scaling * geom.dy + geom.dy / 2,
                    label="inj wells",
                    marker="^",
                    markersize=10,
                    c="red",
                    linestyle="none",
                )

    for format in ["png", "pdf"]:
        plotter.fig.savefig(
            str(fig_save_path.joinpath(f"{fname}.{format}")), format=format
        )
