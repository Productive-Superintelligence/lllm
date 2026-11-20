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
import time
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
        """The system prompt, if the dialog has one.

        Some providers' requests (OpenAI's, for example) carry it with the
        ``developer`` role instead of ``system``.
        """

        for message in self.messages:
            if message.get("role") in ("system", "developer"):
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
    in ``requests``, so tests can check what the model was asked. Calls to
    chat completions (the default) and to OpenAI's Responses API (agents
    with ``api_type="response"``) are both answered.
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
            self._saved = (invoker_module.completion_api, invoker_module.responses_api)
            invoker_module.completion_api = self._complete
            invoker_module.responses_api = self._respond
            FunctionModel._active.append(self)
        return self

    def __exit__(self, *exc_info: Any) -> None:
        from lllm.runtimes.native.invokers import litellm as invoker_module

        with FunctionModel._lock:
            invoker_module.completion_api, invoker_module.responses_api = self._saved
            FunctionModel._active.remove(self)

    def _ask(self, model: str, messages: list[dict[str, Any]], tools: Any, response_format: Any = None) -> Reply:
        request = ModelRequest(
            model=model,
            messages=messages,
            tools=[tool["function"]["name"] for tool in tools or [] if "function" in tool],
            response_format=response_format,
        )
        self.requests.append(request)
        return self.function(request)

    def _complete(self, *, model: str, messages: list[dict[str, Any]], tools: Any = None, **kwargs: Any) -> Any:
        import litellm

        reply = self._ask(model, messages, tools, kwargs.get("response_format"))
        calls = _tool_calls(reply)
        stream = bool(kwargs.get("stream"))
        if calls is None:
            return litellm.completion(
                model=model,
                messages=messages,
                mock_response=str(reply),
                stream=stream,
            )
        tool_calls = [
            {
                "id": f"call_{uuid.uuid4().hex[:12]}",
                "type": "function",
                "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
            }
            for call in calls
        ]
        if stream:  # an agent with a stream handler
            return _tool_call_stream(model, tool_calls)
        response = litellm.completion(model=model, messages=messages, mock_tool_calls=tool_calls)
        response.choices[0].finish_reason = "tool_calls"
        return response

    def _respond(self, *, model: str, input: Any, tools: Any = None, **kwargs: Any) -> Any:
        """Answer a Responses API call (an agent with ``api_type="response"``)."""

        messages = [{"role": "user", "content": input}] if isinstance(input, str) else list(input)
        reply = self._ask(model, messages, tools)
        response = _responses_api_response(model, reply)
        return _responses_api_stream(response) if kwargs.get("stream") else response


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


def _tool_call_stream(model: str, tool_calls: list[dict[str, Any]]) -> Any:
    """The stream chunks of a tool-call reply (LiteLLM's mock streams text only)."""

    from litellm.types.utils import (
        ChatCompletionDeltaToolCall,
        Delta,
        Function,
        ModelResponseStream,
        StreamingChoices,
    )

    deltas = [
        ChatCompletionDeltaToolCall(
            id=call["id"], type="function", index=index, function=Function(**call["function"])
        )
        for index, call in enumerate(tool_calls)
    ]
    first = Delta(role="assistant", content=None, tool_calls=deltas)
    yield ModelResponseStream(model=model, choices=[StreamingChoices(index=0, delta=first)])
    done = StreamingChoices(index=0, delta=Delta(), finish_reason="tool_calls")
    yield ModelResponseStream(model=model, choices=[done])


def _responses_api_response(model: str, reply: Reply) -> Any:
    """A Responses API response, with the same placeholder usage and cost
    as LiteLLM's mock chat completions."""

    import litellm
    from litellm.types.llms.openai import ResponsesAPIResponse

    calls = _tool_calls(reply)
    if calls is None:
        output = [
            {
                "type": "message",
                "id": f"msg_{uuid.uuid4().hex[:12]}",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": str(reply), "annotations": []}],
            }
        ]
    else:
        output = [
            {
                "type": "function_call",
                "id": f"fc_{uuid.uuid4().hex[:12]}",
                "call_id": f"call_{uuid.uuid4().hex[:12]}",
                "name": call.name,
                "arguments": json.dumps(call.arguments),
                "status": "completed",
            }
            for call in calls
        ]
    response = ResponsesAPIResponse.model_validate(
        {
            "id": f"resp_{uuid.uuid4().hex[:12]}",
            "object": "response",
            "created_at": int(time.time()),
            "status": "completed",
            "model": model,
            "output": output,
            "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
        }
    )
    try:
        cost = litellm.cost_per_token(model=model, prompt_tokens=10, completion_tokens=20)
        response._hidden_params["response_cost"] = sum(cost)
    except Exception:  # a model LiteLLM has no price for
        pass
    return response


def _responses_api_stream(response: Any) -> Any:
    """The stream events of a Responses API reply: its text, then the response."""

    from litellm.types.llms.openai import OutputTextDeltaEvent, ResponseCompletedEvent

    for index, item in enumerate(response.output):
        if item.type == "message":
            for part, content in enumerate(item.content):
                yield OutputTextDeltaEvent(
                    type="response.output_text.delta",
                    item_id=item.id,
                    output_index=index,
                    content_index=part,
                    delta=content.text,
                )
    yield ResponseCompletedEvent(type="response.completed", response=response)


def _tool_calls(reply: Reply) -> list[ToolCall] | None:
    if isinstance(reply, ToolCall):
        return [reply]
    if isinstance(reply, (list, tuple)) and reply and all(isinstance(item, ToolCall) for item in reply):
        return list(reply)
    return None
