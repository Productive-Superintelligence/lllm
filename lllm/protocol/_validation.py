"""Private validation helpers for protocol boundary fields."""

from __future__ import annotations

import re
from collections.abc import Mapping
from copy import deepcopy
from typing import Any


def copy_boundary_value(value: Any) -> Any:
    """Return an owned copy of values crossing public LLLM boundaries.

    A mapping becomes a dict, and a list, tuple, set, or frozenset stays
    one, with its items copied the same way; anything else is deep-copied.
    """

    return _copy_nested(value, None)


# Containers copied item by item, each into this plain type; a subclass of
# one becomes the plain type.
_CONTAINER_KINDS = {dict: dict, list: list, tuple: tuple, set: set, frozenset: frozenset}
# Values deepcopy returns as they are.
_ATOMIC = frozenset({str, int, float, bool, type(None), bytes})


def _container_kind(value: Any) -> Any:
    kind = _CONTAINER_KINDS.get(type(value))
    if kind is not None:
        return kind
    if isinstance(value, Mapping):
        return dict
    for kind in (list, tuple, set, frozenset):
        if isinstance(value, kind):
            return kind
    return None


def _copy_nested(value: Any, label: str | None) -> Any:
    """Copy containers into plain dicts, lists, tuples, sets, and frozensets.

    Without a ``label`` this is ``copy_boundary_value``; with one, mapping
    keys must be strings, and a ``TypeError`` names the label of the mapping
    with the bad key (``label.key`` for one nested under ``key``). Shared
    items are copied separately, and anything that is not a container is
    deep-copied. The value is walked with an explicit stack rather than
    recursion, so a deeply nested value cannot overflow a small thread
    stack; a value that contains itself raises ``RecursionError``, as the
    recursive copy did.
    """

    kind = _container_kind(value)
    if kind is None:
        return deepcopy(value)
    # A frame: the container, its kind, an iterator over its children, the
    # copies so far, its label, and the key of the child being copied.
    stack = [[value, kind, _children(value, kind), [], label, None]]
    copying = {id(value)}  # the containers on the stack
    while True:
        frame = stack[-1]
        source, kind, children, copies, frame_label, _ = frame
        for entry in children:
            if kind is dict:
                key, child = entry
                if frame_label is not None and not isinstance(key, str):
                    raise TypeError(f"{frame_label} keys must be strings.")
            else:
                key, child = None, entry
            if type(child) in _ATOMIC:
                copied = child
            else:
                child_kind = _container_kind(child)
                if child_kind is None:
                    copied = deepcopy(child)
                elif not child:  # nothing in it to walk
                    copied = child_kind()
                elif id(child) in copying:
                    raise RecursionError("cannot copy a value that contains itself")
                else:
                    frame[5] = key
                    if kind is dict and frame_label is not None:
                        child_label: str | None = f"{frame_label}.{key}"
                    else:
                        child_label = frame_label
                    copying.add(id(child))
                    stack.append(
                        [child, child_kind, _children(child, child_kind), [], child_label, None]
                    )
                    break
            copies.append((key, copied) if kind is dict else copied)
        else:
            stack.pop()
            copying.discard(id(source))
            copied = kind(copies)
            if not stack:
                return copied
            parent = stack[-1]
            parent[3].append((parent[5], copied) if parent[1] is dict else copied)


def _children(value: Any, kind: Any) -> Any:
    return iter(value.items()) if kind is dict else iter(value)


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


_SCHEMA_METADATA_KEYS = frozenset({"schema", "input_schema", "output_schema"})


def is_sensitive_metadata_key(key: object) -> bool:
    """Whether a metadata key names a secret (api keys, passwords, tokens, ...)."""

    return _is_sensitive_metadata_key(key, match_token_part=False)


def is_public_sensitive_metadata_key(key: object) -> bool:
    """Stricter check for public payloads: any ``token`` word part counts."""

    return _is_sensitive_metadata_key(key, match_token_part=True)


def is_schema_metadata_key(key: object) -> bool:
    return isinstance(key, str) and normalize_metadata_key(key) in _SCHEMA_METADATA_KEYS


def _is_sensitive_metadata_key(key: object, *, match_token_part: bool) -> bool:
    if not isinstance(key, str):
        return False
    normalized = normalize_metadata_key(key)
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
    if compact.endswith(("authorization", "credential", "credentials", "privatekey")):
        return True
    if "password" in parts or "secret" in parts:
        return True
    if compact.endswith(("password", "secret")):
        return True
    if "cookie" in parts:
        return True
    if compact == "cookie" or compact.endswith("cookie"):
        return True
    if match_token_part and "token" in parts:
        return True
    if normalized == "token" or normalized.endswith("_token"):
        return True
    if compact == "token" or compact.endswith("token"):
        return True
    return False


def normalize_metadata_key(key: str) -> str:
    with_word_breaks = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", key)
    with_word_breaks = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", with_word_breaks)
    return re.sub(r"[^a-z0-9]+", "_", with_word_breaks.lower()).strip("_")


def optional_mapping_value(label: str, value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    return mapping_value(label, value)


def mapping_value(label: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping.")
    return _copy_string_keyed_mapping(label, value)


def mapping_field_value(label: str, value: Any) -> dict[str, Any]:
    try:
        return mapping_value(label, value)
    except TypeError as exc:
        raise ValueError(str(exc)) from exc


def mapping_sequence_value(
    label: str,
    value: Any,
    *,
    allow_tuple: bool = False,
) -> list[dict[str, Any]]:
    accepted = (list, tuple) if allow_tuple else list
    if not isinstance(value, accepted):
        raise TypeError(f"{label} must be a list.")
    return _copy_string_keyed_mapping_sequence(label, list(value))


def mapping_sequence_field_value(
    label: str,
    value: Any,
    *,
    allow_tuple: bool = False,
) -> list[dict[str, Any]]:
    try:
        return mapping_sequence_value(label, value, allow_tuple=allow_tuple)
    except TypeError as exc:
        raise ValueError(str(exc)) from exc


def _copy_string_keyed_mapping(
    label: str,
    value: Mapping[Any, Any],
) -> dict[str, Any]:
    return _copy_nested(value, label)


def _copy_string_keyed_mapping_sequence(
    label: str,
    value: list[Any] | tuple[Any, ...],
) -> list[dict[str, Any]]:
    copied: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise TypeError(f"{label} must contain mappings.")
        copied.append(_copy_string_keyed_mapping(f"{label}[{index}]", item))
    return copied


def _copy_string_keyed_value(label: str, value: Any) -> Any:
    return _copy_nested(value, label)


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
    return _copy_nested(value, label)


def _copy_metadata_value(label: str, value: Any) -> Any:
    return _copy_nested(value, label)


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
        or any(ch in text for ch in "/:\\;")
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
        or any(ch in text for ch in "/:\\;")
    ):
        raise ValueError(
            f"{label} must be a non-empty token without whitespace, "
            "percent escapes, or path separators."
        )
    return text
