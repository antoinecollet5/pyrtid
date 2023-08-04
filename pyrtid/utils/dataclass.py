"""Provide utilities for dataclasses."""
import copy
from dataclasses import field
from typing import Any


def default_field(obj) -> Any:
    """Helper to set a default value for a dataclass field."""
    return field(default_factory=lambda: copy.copy(obj))
