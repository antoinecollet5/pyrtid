# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 Antoine COLLET

"""Provide utilities for dataclasses."""

import copy
from dataclasses import field
from typing import Any


def default_field(obj) -> Any:
    """
    Set a default value for a dataclass field (copied for each instance).

    Parameters
    ----------
    obj : Any
        Object used as default value; a shallow copy is made for each instance.

    Returns
    -------
    dataclasses.Field
        The dataclass field with a ``default_factory`` returning a copy of ``obj``.
    """
    return field(default_factory=lambda: copy.copy(obj))
