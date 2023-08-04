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

from .obs_vs_simu import plot_observed_vs_simulated

__all__ = [
    "plot_observed_vs_simulated",
]
