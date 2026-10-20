# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 Antoine COLLET

"""
Reactive transport inversion with the adjoint state method.

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

"""

import warnings

# Make scooby a soft dependency:
try:
    from scooby import Report as ScoobyReport
except ImportError:

    class ScoobyReport:
        """Placeholder for :class:`scooby.Report` when scooby is not installed."""

        def __init__(self, *args, **kwargs):
            """Warn that scooby is required to build a report."""
            message = (
                "\n  *ERROR*: `Report` requires `scooby`."
                "\n           Install it via `pip install scooby` or"
                "\n           `conda install -c conda-forge scooby`."
                "\n           `Note that python >= 3.10 is required!\n"
            )
            warnings.warn(message, stacklevel=2)


from pyrtid import forward, inverse, utils
from pyrtid.__about__ import __author__, __email__, __version__


class Report(ScoobyReport):  # ty:ignore[unsupported-base]
    """Report the versions of pyrtid, python and the main dependencies (scooby)."""

    def __init__(
        self, additional=None, ncol: int = 3, text_width: int = 80, sort: bool = False
    ):
        """
        Initialize a scooby.Report instance.

        Parameters
        ----------
        additional : list of str or module, optional
            Additional packages (or modules) to include in the report. The default is
            None.
        ncol : int, optional
            Number of columns of the report. The default is 3.
        text_width : int, optional
            Text width of the report. The default is 80.
        sort : bool, optional
            Whether to sort the packages alphabetically. The default is False.
        """
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
            "quickpaver",
            "inv_toolbox",
            "iterative_ensemble_smoother",
            "lbfgsb",
            "pypcga",
            "pyesmda",
            "gstools",
        ]

        # Optional packages.
        optional = []
        super().__init__(
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
    "Report",
]
