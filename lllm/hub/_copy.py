"""Private copy helpers for LLLM hub boundary values."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any


def copy_boundary_value(value: Any) -> Any:
    """Return an owned copy of values crossing public LLLM hub boundaries."""
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


def optional_metadata_mapping_value(label: str, value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    return metadata_mapping_value(label, value)


def metadata_mapping_value(label: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping.")
    return _copy_metadata_mapping(label, value)


def metadata_field_value(label: str, value: Any) -> dict[str, Any]:
    try:
        return metadata_mapping_value(label, value)
    except TypeError as exc:
        raise ValueError(str(exc)) from exc


def mapping_field_value(label: str, value: Any) -> dict[str, Any]:
    try:
        return mapping_value(label, value)
    except TypeError as exc:
        raise ValueError(str(exc)) from exc


def mapping_value(label: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping.")
    return _copy_string_keyed_mapping(label, value)


def mapping_sequence_field_value(label: str, value: Any) -> list[dict[str, Any]]:
    try:
        return mapping_sequence_value(label, value)
    except TypeError as exc:
        raise ValueError(str(exc)) from exc


def mapping_sequence_value(label: str, value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{label} must be a list.")
    copied: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise TypeError(f"{label} must contain mappings.")
        copied.append(_copy_string_keyed_mapping(f"{label}[{index}]", item))
    return copied


def _copy_string_keyed_mapping(
    label: str,
    value: Mapping[Any, Any],
) -> dict[str, Any]:
    copied: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError(f"{label} keys must be strings.")
        copied[key] = _copy_string_keyed_value(f"{label}.{key}", item)
    return copied


def _copy_string_keyed_value(label: str, value: Any) -> Any:
    if isinstance(value, Mapping):
        return _copy_string_keyed_mapping(label, value)
    if isinstance(value, list):
        return [_copy_string_keyed_value(label, item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_string_keyed_value(label, item) for item in value)
    if isinstance(value, set):
        return {_copy_string_keyed_value(label, item) for item in value}
    if isinstance(value, frozenset):
        return frozenset(_copy_string_keyed_value(label, item) for item in value)
    return deepcopy(value)


def _copy_metadata_mapping(label: str, value: Mapping[Any, Any]) -> dict[str, Any]:
    copied: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError(f"{label} keys must be strings.")
        copied[key] = _copy_metadata_value(f"{label}.{key}", item)
    return copied


def _copy_metadata_value(label: str, value: Any) -> Any:
    if isinstance(value, Mapping):
        return _copy_metadata_mapping(label, value)
    if isinstance(value, list):
        return [_copy_metadata_value(label, item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_metadata_value(label, item) for item in value)
    if isinstance(value, set):
        return {_copy_metadata_value(label, item) for item in value}
    if isinstance(value, frozenset):
        return frozenset(_copy_metadata_value(label, item) for item in value)
    return deepcopy(value)
