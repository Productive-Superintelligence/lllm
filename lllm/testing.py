"""Run native agents without calling a model provider.

Inside ``with ScriptedModel([...]):`` or ``with FunctionModel(fn):``, every
model call a native agent makes is answered locally. Everything else is the
real code path: the dialog is converted to messages, replies go through the
prompt's parser and ``format=``, parse failures are retried, and tool calls
run your tools. Use it in unit tests, and to try a program before you have
an API key::

    from lllm import Tactic
    from lllm.testing import ScriptedModel

    with ScriptedModel(["Paris."]):
        print(Tactic.quick("What is the capital of France?").content)
    #> Paris.

Requires ``lllm-core[native]`` (LiteLLM builds the response objects).
"""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence, Union

__all__ = ["FunctionModel", "ModelRequest", "ScriptedModel", "ToolCall"]


@dataclass(frozen=True)
class ToolCall:
    """A reply that calls a tool: ``ToolCall("search", {"query": "lllm"})``."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


Reply = Union[str, ToolCall, Sequence[ToolCall]]


@dataclass
class ModelRequest:
    """What the model was asked: the messages, the tools on offer, and the format."""

    model: str
    messages: list[dict[str, Any]]
    tools: list[str]
    response_format: Any = None

    @property
    def last_message(self) -> dict[str, Any]:
        return self.messages[-1] if self.messages else {}

    @property
    def text(self) -> str:
        """The content of the last message, as text."""

        content = self.last_message.get("content")
        if isinstance(content, list):  # multimodal parts
            return " ".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
        return "" if content is None else str(content)

    @property
    def system(self) -> str:
        """The system prompt, if the dialog has one."""

        for message in self.messages:
            if message.get("role") == "system":
                return str(message.get("content") or "")
        return ""

    def tool_results(self) -> list[str]:
        """Contents of the tool results since the last user message."""

        results: list[str] = []
        for message in reversed(self.messages):
            if message.get("role") == "user":
                break
            if message.get("role") == "tool":
                results.append(str(message.get("content") or ""))
        return list(reversed(results))


class FunctionModel:
    """Answer every model call with ``function(request)``.

    The function receives a :class:`ModelRequest` and returns the reply
    text, a :class:`ToolCall`, or a list of tool calls. Every request is kept
    in ``requests``, so tests can check what the model was asked.
    """

    _lock = threading.RLock()
    _active: list["FunctionModel"] = []

    def __init__(self, function: Callable[[ModelRequest], Reply]) -> None:
        self.function = function
        self.requests: list[ModelRequest] = []
        self._saved: Any = None

    def __enter__(self) -> "FunctionModel":
        from lllm.runtimes.native.invokers import litellm as invoker_module

        with FunctionModel._lock:
            self._saved = invoker_module.completion_api
            invoker_module.completion_api = self._complete
            FunctionModel._active.append(self)
        return self

    def __exit__(self, *exc_info: Any) -> None:
        from lllm.runtimes.native.invokers import litellm as invoker_module

        with FunctionModel._lock:
            invoker_module.completion_api = self._saved
            FunctionModel._active.remove(self)

    def _complete(self, *, model: str, messages: list[dict[str, Any]], tools: Any = None, **kwargs: Any) -> Any:
        import litellm

        request = ModelRequest(
            model=model,
            messages=messages,
            tools=[tool["function"]["name"] for tool in tools or [] if "function" in tool],
            response_format=kwargs.get("response_format"),
        )
        self.requests.append(request)
        reply = self.function(request)
        calls = _tool_calls(reply)
        if calls is None:
            return litellm.completion(
                model=model,
                messages=messages,
                mock_response=str(reply),
                stream=bool(kwargs.get("stream")),
            )
        response = litellm.completion(
            model=model,
            messages=messages,
            mock_tool_calls=[
                {
                    "id": f"call_{uuid.uuid4().hex[:12]}",
                    "type": "function",
                    "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
                }
                for call in calls
            ],
        )
        response.choices[0].finish_reason = "tool_calls"
        return response


class ScriptedModel(FunctionModel):
    """Answer with ``replies`` in order; once they run out, the last one repeats."""

    def __init__(self, replies: Iterable[Reply]) -> None:
        self.replies = list(replies)
        if not self.replies:
            raise ValueError("ScriptedModel needs at least one reply.")
        self._next = 0
        super().__init__(self._reply)

    def _reply(self, request: ModelRequest) -> Reply:
        reply = self.replies[min(self._next, len(self.replies) - 1)]
        self._next += 1
        return reply


def _tool_calls(reply: Reply) -> list[ToolCall] | None:
    if isinstance(reply, ToolCall):
        return [reply]
    if isinstance(reply, (list, tuple)) and reply and all(isinstance(item, ToolCall) for item in reply):
        return list(reply)
    return None
