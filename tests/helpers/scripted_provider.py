from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from lllm.core.const import APITypes, Roles
from lllm.core.models import Message
from lllm.invokers.base import BaseInvoker


class ScriptedInvoker(BaseInvoker):
    """Test helper invoker that replays scripted responses and runs prompt parsers."""

    def __init__(self, scripts: Iterable[Dict[str, Any]]):
        self._queue: List[Dict[str, Any]] = list(scripts)
        self.call_count = 0
        self.errors: List[List[Exception]] = []

    def call(
        self,
        dialog,
        prompt,
        model,
        model_args=None,
        parser_args: Optional[Dict[str, Any]] = None,
        responder: str = "assistant",
        extra: Optional[Dict[str, Any]] = None,
        api_type: APITypes = APITypes.COMPLETION,
    ):
        if not self._queue:
            raise AssertionError("ScriptedInvoker received more calls than scripted responses")

        script = self._queue.pop(0)
        self.call_count += 1

        content = script.get("content", "")
        role = script.get("role")
        function_calls = script.get("function_calls", [])
        errors = []
        parsed = script.get("parsed", {})

        if prompt.parser is not None and "parsed" not in script:
            try:
                parsed = prompt.parse(content, **(parser_args or {}))
            except Exception as exc:
                errors.append(exc)
                parsed = {"raw": content}

        self.errors.append(errors.copy())

        message_role = role or (Roles.TOOL_CALL if function_calls else Roles.ASSISTANT)
        message_api_type = script.get("api_type", api_type)

        return Message(
            role=message_role,
            content=content,
            name=responder,
            function_calls=function_calls,
            parsed=parsed,
            execution_errors=errors,
            model=model,
            model_args=model_args or {},
            extra=extra or {},
            api_type=message_api_type,
        )

    def stream(self, *args, **kwargs):
        raise NotImplementedError("ScriptedInvoker does not support streaming")
