"""Call metadata passed through tactic and service boundaries."""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CallContext(BaseModel):
    """Metadata supplied to one tactic invocation.

    This is request metadata, not runtime configuration. Runtime-specific
    settings stay with the runtime adapter or user-owned runtime object.
    """

    model_config = ConfigDict(extra="allow")

    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    caller: str | None = None
    trace_id: str | None = None
    span_id: str | None = None
    package_ref: str | None = None
    service_ref: str | None = None
    tactic_ref: str | None = None
    endpoint: str | None = None
    tags: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def call_id(self) -> str:
        """Compatibility alias for older boundary traces."""

        return self.request_id
