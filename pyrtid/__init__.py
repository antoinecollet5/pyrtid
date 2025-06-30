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

from pyrtid import forward, inverse, plot, regularization, utils
from pyrtid.__about__ import __author__, __email__, __version__

__all__ = [
    "__version__",
    "__email__",
    "__author__",
    "forward",
    "inverse",
    "utils",
    "plot",
    "regularization",
]
