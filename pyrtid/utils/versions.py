"""Utilities to parse packages versions."""
import logging

import gstools
import matplotlib
import nested_grid_plotter
import numdifftools
import numpy as np
import pyesmda
import pyPCGA
import scipy
import sksparse
import stochopy

from pyrtid.__about__ import __version__


def show_versions() -> None:
    """Show the versions of all packages used by pyrtid."""

    logging.info(f"Current version = {__version__}\n")
    logging.info("Used packages version:\n")
    logging.info("iterative_ensemble_smoother = 0.1.1")  # todo update the library
    logging.info(f"gstools                     = {gstools.__version__}")
    logging.info(f"matplotlib                  = {matplotlib.__version__}")
    logging.info(f"nested_grid_plotter         = {nested_grid_plotter.__version__}")
    logging.info(f"numdiftools                 = {numdifftools.__version__}")
    logging.info(f"numpy                       = {np.__version__}")
    logging.info(f"pyesmda                     = {pyesmda.__version__}")
    logging.info(f"pypcga                      = {pyPCGA.__version__}")
    logging.info(f"scipy                       = {scipy.__version__}")
    logging.info(f"sksparse                    = {sksparse.__version__}")
    logging.info(f"stochopy                    = {stochopy.__version__}")
