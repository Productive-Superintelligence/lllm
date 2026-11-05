"""Shared metadata boundary helpers."""

from __future__ import annotations

import re

_SCHEMA_METADATA_KEYS = frozenset({"schema", "input_schema", "output_schema"})


def is_schema_metadata_key(key: object) -> bool:
    return (
        isinstance(key, str)
        and normalize_metadata_key(key) in _SCHEMA_METADATA_KEYS
    )


def is_public_sensitive_metadata_key(key: object) -> bool:
    return _is_sensitive_metadata_key(key, match_token_part=True)


def is_sensitive_metadata_key(key: object) -> bool:
    return _is_sensitive_metadata_key(key, match_token_part=False)


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
    if (
        "authorization" in parts
        or "credential" in parts
        or "credentials" in parts
    ):
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
