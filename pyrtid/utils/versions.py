"""Utilities to parse packages versions."""

import logging

import gstools
import lbfgsb
import matplotlib
import nested_grid_plotter
import numdifftools
import numpy as np
import pyesmda
import scipy
import sksparse
import stochopy

from pyrtid.__about__ import __version__


def show_versions(logger: logging.Logger) -> None:
    """Show the versions of all packages used by pyrtid."""

    logger.info(f"Current version = {__version__}\n")
    logger.info("Used packages version:\n")
    logger.info("iterative_ensemble_smoother = 0.1.1")  # todo update the library
    logger.info(f"gstools                     = {gstools.__version__}")
    logger.info(f"matplotlib                  = {matplotlib.__version__}")
    logger.info(f"nested_grid_plotter         = {nested_grid_plotter.__version__}")
    logger.info(f"numdiftools                 = {numdifftools.__version__}")
    logger.info(f"numpy                       = {np.__version__}")
    logger.info(f"pyesmda                     = {pyesmda.__version__}")
    # logger.info(f"pypcga                      = {pyPCGA.__version__}")
    logger.info(f"lbfgsb                      = {lbfgsb.__version__}")
    logger.info(f"scipy                       = {scipy.__version__}")
    logger.info(f"sksparse                    = {sksparse.__version__}")
    logger.info(f"stochopy                    = {stochopy.__version__}")
