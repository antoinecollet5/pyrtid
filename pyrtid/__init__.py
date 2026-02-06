"""
Purpose
=======

**pyrtid** is an open-source, pure python, and object-oriented library that provides
a user friendly implementation of inversion for reactive transport code.

Submodules
==========

.. autosummary::
    forward
    inverse
    utils
    plot
    regularization

"""

import scooby
from pyrtid import forward, inverse, plot, regularization, utils
from pyrtid.__about__ import __author__, __email__, __version__


class Report(scooby.Report):
    def __init__(self, additional=None, ncol=3, text_width=80, sort=False):
        """Initiate a scooby.Report instance."""

        # Mandatory packages.
        core = [
            "pyrtid",
            "numpy",
            "scipy",
            "matplotlib",
            "joblib",
            "typing_extensions",
            "nested_grid_plotter",
            "stochopy",
            "scooby",
            "iterative_ensemble_smoother",
            "lbfgsb",
            "gstools",
        ]

        # Optional packages.
        optional = ["suitesparse", "scikit-sparse"]

        scooby.Report.__init__(
            self,
            additional=additional,
            core=core,
            optional=optional,
            ncol=ncol,
            text_width=text_width,
            sort=sort,
        )


__all__ = [
    "__version__",
    "__email__",
    "__author__",
    "forward",
    "inverse",
    "utils",
    "plot",
    "regularization",
    "Report",
]
