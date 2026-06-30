"""Call metadata passed through tactic and service boundaries."""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictStr, model_validator

from ._validation import copy_boundary_value, token_value


class CallContext(BaseModel):
    """Metadata supplied to one tactic invocation.

    This is request metadata, not runtime configuration. Runtime-specific
    settings stay with the runtime adapter or user-owned runtime object.
    """

    model_config = ConfigDict(extra="allow")

    request_id: StrictStr = Field(default_factory=lambda: str(uuid.uuid4()))
    caller: StrictStr | None = None
    trace_id: StrictStr | None = None
    span_id: StrictStr | None = None
    package_ref: StrictStr | None = None
    service_ref: StrictStr | None = None
    tactic_ref: StrictStr | None = None
    endpoint: StrictStr | None = None
    tags: dict[StrictStr, StrictStr] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def call_id(self) -> str:
        """Compatibility alias for older boundary traces."""

        return self.request_id

    def model_post_init(self, __context: Any) -> None:
        self.metadata = copy_boundary_value(self.metadata)
        self.tags = copy_boundary_value(self.tags)

    @model_validator(mode="after")
    def _validate_identifiers(self) -> "CallContext":
        token_value(self.request_id, "request_id")
        if self.trace_id is not None:
            token_value(self.trace_id, "trace_id")
        if self.span_id is not None:
            token_value(self.span_id, "span_id")
        return self
