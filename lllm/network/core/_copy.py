"""Private copy helpers for LLLM network boundary values."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any


def copy_boundary_value(value: Any) -> Any:
    """Return an owned copy of values crossing public LLLM network boundaries."""

    if isinstance(value, Mapping):
        return {key: copy_boundary_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [copy_boundary_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(copy_boundary_value(item) for item in value)
    if isinstance(value, set):
        return {copy_boundary_value(item) for item in value}
    if isinstance(value, frozenset):
        return frozenset(copy_boundary_value(item) for item in value)
    return deepcopy(value)
