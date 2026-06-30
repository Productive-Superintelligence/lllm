"""Private validation helpers for protocol boundary fields."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any


def copy_boundary_value(value: Any) -> Any:
    """Return an owned copy of values crossing public LLLM boundaries."""

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


def optional_mapping_value(label: str, value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping.")
    return {key: copy_boundary_value(item) for key, item in value.items()}


def text_value(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string.")
    return value


def optional_text_value(value: Any, label: str) -> str | None:
    if value is None:
        return None
    return text_value(value, label)


def path_segment_value(value: Any, label: str) -> str:
    text = text_value(value, label)
    if (
        not text.strip()
        or text in {".", ".."}
        or "%" in text
        or any(ch in text for ch in "/:\\")
    ):
        raise ValueError(
            f"{label} must be a non-empty name without percent escapes "
            "or path separators."
        )
    return text


def token_value(value: Any, label: str) -> str:
    text = text_value(value, label)
    if (
        not text.strip()
        or text in {".", ".."}
        or "%" in text
        or any(ch.isspace() for ch in text)
        or any(ch in text for ch in "/:\\")
    ):
        raise ValueError(
            f"{label} must be a non-empty token without whitespace, "
            "percent escapes, or path separators."
        )
    return text
