# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 Antoine COLLET

"""
Provides the derivatives of the equations F with respect to the state variables u.

This is required to build the left and right hand sides in the forward sensitivity
method (FSM).
"""

from __future__ import annotations

import numpy as np

from pyrtid.forward.models import ForwardModel
from pyrtid.utils import NDArrayFloat


def dFhdhv(
    fwd_model: ForwardModel, time_index: int, vecs: NDArrayFloat
) -> NDArrayFloat:
    """
    Return the product of dFh/dh with an ensemble of vectors v.

    Parameters
    ----------
    fwd_model : ForwardModel
        _description_
    time_index : int
        _description_
    vecs : NDArrayFloat
        _description_

    Returns
    -------
    NDArrayFloat
        _description_
    """
    # step 1 assert size of vecs

    return vecs


def dFpdrhov(
    fwd_model: ForwardModel, time_index: int, vecs: NDArrayFloat
) -> NDArrayFloat:
    # Case one = saturated flow = no impact of density
    if not fwd_model.fl_model.is_gravity:
        return np.zeros_like(fwd_model.fl_model.lpressure[0])
    # Case two = density flow = impact of density
    return vecs


def dFUdrhov(
    fwd_model: ForwardModel, time_index: int, vecs: NDArrayFloat
) -> NDArrayFloat:
    # Case one = saturated flow = no impact of density
    if not fwd_model.fl_model.is_gravity:
        return np.zeros_like(fwd_model.fl_model.lpressure[0])
    # Case two = density flow = impact of density

    # HERE: TODO

    return vecs
