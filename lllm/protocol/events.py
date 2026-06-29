"""Runtime-neutral event shapes for streaming tactics."""

from __future__ import annotations

import time
import uuid
from copy import deepcopy
from typing import Any

from pydantic import BaseModel, Field


class TacticEvent(BaseModel):
    """A small event emitted by a streaming tactic."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = Field(default_factory=time.time)
    kind: str = "message"
    data: Any = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        self.data = deepcopy(self.data)
        self.metadata = deepcopy(self.metadata)

    @classmethod
    def result(cls, value: Any, **metadata: Any) -> "TacticEvent":
        return cls(kind="result", data=value, metadata=metadata)

    @classmethod
    def error(cls, message: str, **metadata: Any) -> "TacticEvent":
        return cls(kind="error", data={"message": message}, metadata=metadata)
