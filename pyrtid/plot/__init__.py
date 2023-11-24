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

from .plt_gradient import plot_2d_grad_res_adj_vs_fd
from .plt_obs_vs_simu import plot_observed_vs_simulated

__all__ = ["plot_observed_vs_simulated", "plot_2d_grad_res_adj_vs_fd"]
