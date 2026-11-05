"""Public LLLM shell data models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator


ResourceKind = Literal[
    "tactic",
    "channel",
    "service",
    "schema",
    "package",
    "run",
    "config",
    "doc",
    "example",
    "snapshot",
    "asset",
    "custom",
]


class StrategyResource(BaseModel):
    """A module or resource encapsulated by a strategy."""

    model_config = ConfigDict(frozen=True)

    name: StrictStr
    kind: ResourceKind
    ref: StrictStr | None = None
    entrypoint: StrictStr | None = None
    description: StrictStr = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        return _segment(value, "resource.name")

    @field_validator("metadata")
    @classmethod
    def _validate_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return dict(value)


class StrategyInfo(BaseModel):
    """Public description of a strategy."""

    name: StrictStr
    description: StrictStr = ""
    resources: tuple[StrategyResource, ...] = Field(default_factory=tuple)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        return _segment(value, "strategy.name")


class StrategyRunRequest(BaseModel):
    """Request envelope for strategy calls."""

    input: Any = None
    context: dict[str, Any] | None = None


class StrategyRunResponse(BaseModel):
    """Response envelope for strategy calls."""

    output: Any = None
    strategy: StrictStr
    metadata: dict[str, Any] = Field(default_factory=dict)


def _segment(value: str, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value in {".", ".."}
        or "%" in value
        or any(ch.isspace() for ch in value)
        or any(ch in value for ch in "/:\\")
    ):
        raise ValueError(f"{field_name} must be a non-empty path segment")
    return value
