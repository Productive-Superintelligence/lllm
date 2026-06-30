"""Runtime-neutral event shapes for streaming tactics."""

from __future__ import annotations

import time
import uuid
from typing import Any

from pydantic import BaseModel, Field, StrictStr, model_validator

from ._validation import copy_boundary_value, token_value


class TacticEvent(BaseModel):
    """A small event emitted by a streaming tactic."""

    id: StrictStr = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = Field(default_factory=time.time)
    kind: StrictStr = "message"
    data: Any = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        self.data = copy_boundary_value(self.data)
        self.metadata = copy_boundary_value(self.metadata)

    @model_validator(mode="after")
    def _validate_identity(self) -> "TacticEvent":
        token_value(self.id, "event.id")
        token_value(self.kind, "event.kind")
        return self

    @classmethod
    def result(cls, value: Any, **metadata: Any) -> "TacticEvent":
        return cls(kind="result", data=value, metadata=metadata)

    @classmethod
    def error(cls, message: str, **metadata: Any) -> "TacticEvent":
        return cls(kind="error", data={"message": message}, metadata=metadata)
