"""Call metadata passed through tactic and service boundaries."""

from __future__ import annotations

from copy import deepcopy
import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictStr


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
        self.metadata = deepcopy(self.metadata)
        self.tags = deepcopy(self.tags)
