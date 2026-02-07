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

# Make scooby a soft dependency:
try:
    from scooby import Report as ScoobyReport
except ImportError:

    class ScoobyReport:
        def __init__(self, *args, **kwargs):
            message = (
                "\n  *ERROR*: `Report` requires `scooby`."
                "\n           Install it via `pip install scooby` or"
                "\n           `conda install -c conda-forge scooby`."
                "\n           `Note that python >= 3.10 is required!\n"
            )
            raise ImportError(message)


from pyrtid import forward, inverse, plot, regularization, utils
from pyrtid.__about__ import __author__, __email__, __version__


class Report(ScoobyReport):
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
        ScoobyReport().__init__(
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
