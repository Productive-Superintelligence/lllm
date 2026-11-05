"""Core semantic channel resources."""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr, model_validator

from ._copy import copy_boundary_value

ChannelForm = Literal[
    "log",
    "queue",
    "topic",
    "blackboard",
    "latest-state",
    "artifact-index",
    "pipeline-edge",
    "database-view",
    "feed",
    "time-series",
    "tree",
    "forum",
    "hub",
]


class Channel(BaseModel):
    """A named semantic data interface."""

    model_config = ConfigDict(frozen=True)

    name: StrictStr
    schema_ref: StrictStr | None = Field(default=None, alias="schema")
    form: ChannelForm = "log"
    description: StrictStr = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def schema(self) -> str | None:
        return self.schema_ref

    def model_post_init(self, __context: Any) -> None:
        _set_model_attr(self, "metadata", copy_boundary_value(self.metadata))

    @model_validator(mode="after")
    def _validate_name(self) -> "Channel":
        _validate_segment(self.name, "channel.name")
        return self


class Event(BaseModel):
    """Timestamped semantic record in a channel."""

    id: StrictStr = Field(default_factory=lambda: str(uuid.uuid4()))
    cursor: int | None = None
    channel: StrictStr
    timestamp: float = Field(default_factory=time.time)
    source: StrictStr | None = None
    kind: StrictStr = "event"
    payload: Any = None
    schema_ref: StrictStr | None = Field(default=None, alias="schema")
    metadata: dict[str, Any] = Field(default_factory=dict)
    correlation_id: StrictStr | None = None
    parent_ids: tuple[StrictStr, ...] = Field(default_factory=tuple)

    @property
    def schema(self) -> str | None:
        return self.schema_ref

    def model_post_init(self, __context: Any) -> None:
        _set_model_attr(self, "payload", copy_boundary_value(self.payload))
        _set_model_attr(self, "metadata", copy_boundary_value(self.metadata))

    @model_validator(mode="after")
    def _validate_identity(self) -> "Event":
        _validate_segment(self.id, "event.id")
        _validate_segment(self.channel, "event.channel")
        _validate_token(self.kind, "event.kind")
        if self.correlation_id is not None:
            _validate_token(self.correlation_id, "event.correlation_id")
        for parent_id in self.parent_ids:
            _validate_segment(parent_id, "event.parent_ids")
        return self


class Artifact(BaseModel):
    """Larger payload stored by reference."""

    id: StrictStr = Field(default_factory=lambda: str(uuid.uuid4()))
    channel: StrictStr | None = None
    path: StrictStr
    media_type: StrictStr = "application/octet-stream"
    size: int
    sha256: StrictStr | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    event_ids: tuple[StrictStr, ...] = Field(default_factory=tuple)

    def model_post_init(self, __context: Any) -> None:
        _set_model_attr(self, "metadata", copy_boundary_value(self.metadata))

    @model_validator(mode="after")
    def _validate_identity(self) -> "Artifact":
        _validate_segment(self.id, "artifact.id")
        _validate_optional_segment(self.channel, "artifact.channel")
        for event_id in self.event_ids:
            _validate_segment(event_id, "artifact.event_ids")
        return self


class Snapshot(BaseModel):
    """Latest state or materialized view."""

    name: StrictStr
    channel: StrictStr | None = None
    timestamp: float = Field(default_factory=time.time)
    value: Any = None
    schema_ref: StrictStr | None = Field(default=None, alias="schema")
    source_event_id: StrictStr | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def schema(self) -> str | None:
        return self.schema_ref

    def model_post_init(self, __context: Any) -> None:
        _set_model_attr(self, "value", copy_boundary_value(self.value))
        _set_model_attr(self, "metadata", copy_boundary_value(self.metadata))

    @model_validator(mode="after")
    def _validate_identity(self) -> "Snapshot":
        _validate_segment(self.name, "snapshot.name")
        _validate_optional_segment(self.channel, "snapshot.channel")
        _validate_optional_segment(self.source_event_id, "snapshot.source_event_id")
        return self


class Subscription(BaseModel):
    """Consumer cursor over one channel."""

    id: StrictStr = Field(default_factory=lambda: str(uuid.uuid4()))
    channel: StrictStr
    cursor: int = 0
    consumer: StrictStr | None = None
    batch_size: int = 100
    filters: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        _set_model_attr(self, "filters", copy_boundary_value(self.filters))
        _set_model_attr(self, "metadata", copy_boundary_value(self.metadata))

    @model_validator(mode="after")
    def _validate_identity(self) -> "Subscription":
        _validate_segment(self.id, "subscription.id")
        _validate_segment(self.channel, "subscription.channel")
        if self.consumer is not None:
            _validate_token(self.consumer, "subscription.consumer")
        return self


def _validate_optional_segment(value: str | None, field_name: str) -> None:
    if value is not None:
        _validate_segment(value, field_name)


def _validate_segment(value: str, field_name: str) -> None:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value in {".", ".."}
        or "%" in value
        or any(ch.isspace() for ch in value)
        or any(ch in value for ch in "/:\\")
    ):
        raise ValueError(f"{field_name} must be a non-empty path segment.")


def _validate_token(value: str, field_name: str) -> None:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value in {".", ".."}
        or "%" in value
        or any(ch.isspace() for ch in value)
        or any(ch in value for ch in "/:\\")
    ):
        raise ValueError(f"{field_name} must be a non-empty token.")


def _set_model_attr(model: BaseModel, name: str, value: Any) -> None:
    object.__setattr__(model, name, value)
