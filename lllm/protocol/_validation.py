"""Private validation helpers for protocol boundary fields."""

from __future__ import annotations

import re
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


def public_boundary_value(value: Any) -> Any:
    """Return an owned copy with raw secret-shaped metadata keys omitted."""

    if isinstance(value, Mapping):
        return {
            key: public_boundary_value(item)
            for key, item in value.items()
            if isinstance(key, str) and not is_sensitive_metadata_key(key)
        }
    if isinstance(value, list):
        return [public_boundary_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(public_boundary_value(item) for item in value)
    return copy_boundary_value(value)


def is_sensitive_metadata_key(key: object) -> bool:
    if not isinstance(key, str):
        return False
    normalized = _normalize_metadata_key(key)
    if not normalized:
        return False
    compact = normalized.replace("_", "")
    if normalized.endswith(("_ref", "_refs", "_reference", "_references")):
        return False
    if compact.endswith(("ref", "refs", "reference", "references")):
        return False
    parts = normalized.split("_")
    if "api" in parts and "key" in parts:
        return True
    if compact.endswith("apikey"):
        return True
    if "authorization" in parts or "credential" in parts or "credentials" in parts:
        return True
    if "password" in parts or "secret" in parts:
        return True
    if compact.endswith(("password", "secret")):
        return True
    if "cookie" in parts:
        return True
    if compact == "cookie" or compact.endswith("cookie"):
        return True
    if normalized == "token" or normalized.endswith("_token"):
        return True
    if compact == "token" or compact.endswith("token"):
        return True
    return False


def _normalize_metadata_key(key: str) -> str:
    with_word_breaks = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", key)
    with_word_breaks = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", with_word_breaks)
    return re.sub(r"[^a-z0-9]+", "_", with_word_breaks.lower()).strip("_")


def optional_mapping_value(label: str, value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping.")
    copied: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError(f"{label} keys must be strings.")
        copied[key] = copy_boundary_value(item)
    return copied


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
