"""Pydantic-compatible schema helpers.

The protocol depends on Pydantic and JSON Schema, not on any agent runtime.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, TypeAdapter

from .errors import SchemaError

SchemaRef = Any


def validate_with_schema(value: Any, schema: SchemaRef | None) -> Any:
    """Validate *value* with a Pydantic-compatible schema when one is present."""

    if schema is None or schema is Any:
        return value
    try:
        return TypeAdapter(schema).validate_python(value)
    except Exception as exc:  # pragma: no cover - exact Pydantic text varies
        raise SchemaError(f"Value does not match schema {type_name(schema)}: {exc}") from exc


def export_json_schema(schema: SchemaRef | None) -> dict[str, Any] | None:
    """Return JSON Schema for a Pydantic-compatible schema."""

    if schema is None or schema is Any:
        return None
    try:
        return TypeAdapter(schema).json_schema()
    except Exception as exc:  # pragma: no cover - exact Pydantic text varies
        raise SchemaError(f"Cannot export JSON Schema for {type_name(schema)}: {exc}") from exc


def type_name(schema: SchemaRef | None) -> str | None:
    """Return a readable type name for a schema."""

    if schema is None:
        return None
    if schema is Any:
        return "Any"
    if isinstance(schema, type):
        return schema.__name__
    if isinstance(schema, BaseModel):
        return type(schema).__name__
    name = getattr(schema, "__name__", None)
    if name:
        return str(name)
    return str(schema).replace("typing.", "")
