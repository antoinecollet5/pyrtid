"""
PYRTID submodule providing a set of handy plot tools.

.. currentmodule:: pyrtid.plot

Plot functions
^^^^^^^^^^^^^^
Functions to plot inversion results.

.. autosummary::
   :toctree: _autosummary

   plot_observed_vs_simulated

"""

from pathlib import Path

from pyrtid.plot.config import apply_default_rc_params, register_default_fonts
from pyrtid.plot.plt_gradient import plot_2d_grad_res_adj_vs_fd
from pyrtid.plot.plt_obs_vs_simu import plot_observed_vs_simulated
from pyrtid.plot.plt_percentiles import plot_percentiles

register_default_fonts(Path(__file__).parent.parent.parent.joinpath("fonts"))

__all__ = [
    "apply_default_rc_params",
    "plot_observed_vs_simulated",
    "plot_2d_grad_res_adj_vs_fd",
    "plot_percentiles",
]
