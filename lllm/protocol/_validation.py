"""Private validation helpers for protocol boundary fields."""

from __future__ import annotations

from typing import Any


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
        or any(ch in text for ch in "/:\\")
    ):
        raise ValueError(f"{label} must be a non-empty name without path separators.")
    return text


def token_value(value: Any, label: str) -> str:
    text = text_value(value, label)
    if (
        not text.strip()
        or text in {".", ".."}
        or any(ch.isspace() for ch in text)
        or any(ch in text for ch in "/:\\")
    ):
        raise ValueError(
            f"{label} must be a non-empty token without whitespace or path separators."
        )
    return text
