"""Regression tests for the Pydantic AI runtime adapter fixes (P1-P5)."""

import asyncio
import dataclasses
import inspect
import json
import typing
from datetime import datetime, timezone
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from lllm import SchemaError, Tactic, TacticEvent
from lllm.runtimes import PydanticAITactic, tactic_as_tool
from lllm.runtimes.native import Tactic as NativeTactic
from lllm.services import create_tactic_app


async def collect(iterator):
    return [item async for item in iterator]


def request(app, method, path, **kwargs):
    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.request(method, path, **kwargs)
            return response.status_code, (await response.aread()).decode("utf-8")

    return asyncio.run(run())


def sse_payloads(text):
    return [
        json.loads(line.removeprefix("data: "))
        for line in text.splitlines()
        if line.startswith("data: ")
    ]


def real_pydantic_ai():
    pydantic_ai = pytest.importorskip("pydantic_ai")
    models_test = pytest.importorskip("pydantic_ai.models.test")
    return pydantic_ai.Agent, models_test.TestModel


# -- P1: tactic_as_tool / Tactic.as_tool annotations -----------------------


class AddInput(BaseModel):
    left: int
    right: int = 1


class AddTactic(Tactic[AddInput, int]):
    name = "add-numbers"
    description = "Add two integers."
    input_type = AddInput
    output_type = int

    def __init__(self):
        super().__init__()
        self.calls = []

    def _run(self, input_value, *, context=None):
        self.calls.append(input_value)
        return input_value.left + input_value.right


class LooseTactic(Tactic[Any, Any]):
    name = "loose"

    def _run(self, input_value, *, context=None):
        return input_value


class EchoNativeTactic(NativeTactic):
    name = "echo"
    input_type = str
    output_type = str

    def call(self, task: str) -> str:
        return f"echo: {task}"


def assert_annotations_match_signature(tool):
    signature = inspect.signature(tool)
    expected = {name: param.annotation for name, param in signature.parameters.items()}
    expected["return"] = signature.return_annotation
    # Pydantic AI resolves tool parameter types with get_type_hints().
    assert typing.get_type_hints(tool) == expected


def test_tactic_as_tool_task_mode_annotations_use_input_and_output_types():
    tool = tactic_as_tool(AddTactic())

    assert typing.get_type_hints(tool) == {"task": AddInput, "return": int}
    assert_annotations_match_signature(tool)


def test_tactic_as_tool_kwargs_mode_annotations_use_model_fields():
    tool = tactic_as_tool(AddTactic(), parameter_mode="kwargs")

    assert typing.get_type_hints(tool) == {"left": int, "right": int, "return": int}
    assert_annotations_match_signature(tool)
    assert tool(left=2, right=3) == 5


def test_tactic_as_tool_annotations_fall_back_to_any_for_untyped_tactics():
    task_tool = tactic_as_tool(LooseTactic())
    kwargs_tool = tactic_as_tool(LooseTactic(), parameter_mode="kwargs")

    assert typing.get_type_hints(task_tool) == {"task": Any, "return": Any}
    assert typing.get_type_hints(kwargs_tool) == {"kwargs": Any, "return": Any}
    assert kwargs_tool(a=1) == {"a": 1}


def test_native_tactic_as_tool_annotations_match_signature():
    tool = EchoNativeTactic().as_tool()

    assert typing.get_type_hints(tool) == {"task": str, "return": str}
    assert tool(task="hi") == "echo: hi"


@pytest.mark.parametrize("parameter_mode", ["task", "kwargs"])
def test_tactic_as_tool_registers_and_runs_with_real_pydantic_ai_agent(parameter_mode):
    Agent, TestModel = real_pydantic_ai()
    tactic = AddTactic()
    model = TestModel()
    agent = Agent(model, tools=[tactic_as_tool(tactic, parameter_mode=parameter_mode)])

    agent.run_sync("add some numbers")

    tool_def = model.last_model_request_parameters.function_tools[0]
    assert tool_def.name == "add_numbers"
    assert tool_def.description == "Add two integers."
    assert tool_def.parameters_json_schema["properties"] == {
        "left": {"type": "integer"},
        "right": {"default": 1, "type": "integer"},
    }
    assert tool_def.parameters_json_schema["required"] == ["left"]
    assert len(tactic.calls) == 1
    assert isinstance(tactic.calls[0], AddInput)


def test_native_tactic_as_tool_gives_real_pydantic_ai_agent_a_typed_task():
    Agent, TestModel = real_pydantic_ai()
    model = TestModel()

    Agent(model, tools=[EchoNativeTactic().as_tool()]).run_sync("echo")

    schema = model.last_model_request_parameters.function_tools[0].parameters_json_schema
    assert schema["properties"] == {"task": {"type": "string"}}
    assert schema["required"] == ["task"]


# -- P2: stream/astream/aevents validate input ------------------------------


class BriefInput(BaseModel):
    topic: str
    audience: str = "engineers"


VALIDATED_TASK = '{"topic":"x","audience":"engineers"}'


class StreamResult:
    def __init__(self, chunks):
        self._chunks = chunks

    def stream_output(self):
        yield from self._chunks


class AsyncChunks:
    def __init__(self, chunks):
        self._chunks = chunks

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        for chunk in self._chunks:
            yield chunk


class EventStream:
    def __init__(self, events):
        self._events = iter(events)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._events)
        except StopIteration:
            raise StopAsyncIteration from None


class EventsHandle:
    """Same shape as Pydantic AI's run_stream_events(): a context manager
    that yields a separate async event iterator."""

    def __init__(self, events):
        self._events = events

    async def __aenter__(self):
        return EventStream(self._events)

    async def __aexit__(self, exc_type, exc, tb):
        return None


class TypedStreamAgent:
    """Reads fields from the task, so an unvalidated dict fails."""

    name = "typed-stream"
    output_type = str

    def __init__(self):
        self.tasks = []

    def _chunks(self, task):
        self.tasks.append(task)
        data = json.loads(task) if isinstance(task, str) else dict(task)
        return [data["topic"], data["audience"]]

    def run_sync(self, task, **kwargs):
        return StreamResult(self._chunks(task))

    def run_stream_sync(self, task, **kwargs):
        return StreamResult(self._chunks(task))


class TypedAsyncStreamAgent(TypedStreamAgent):
    def run_stream(self, task, **kwargs):
        return AsyncChunks(self._chunks(task))


class TypedEventsAgent(TypedStreamAgent):
    def run_stream_events(self, task, **kwargs):
        return EventsHandle(self._chunks(task))


class CountingPydanticAITactic(PydanticAITactic):
    validations = 0

    def validate_input(self, value):
        self.validations += 1
        return super().validate_input(value)


def test_stream_validates_input_before_mapping():
    agent = TypedStreamAgent()
    tactic = PydanticAITactic(agent, input_type=BriefInput)

    assert list(tactic.stream({"topic": "x"})) == ["x", "engineers"]
    assert agent.tasks == [VALIDATED_TASK]


@pytest.mark.parametrize("agent_type", [TypedAsyncStreamAgent, TypedStreamAgent])
def test_astream_validates_input_before_mapping(agent_type):
    agent = agent_type()
    tactic = PydanticAITactic(agent, input_type=BriefInput)

    assert asyncio.run(collect(tactic.astream({"topic": "x"}))) == ["x", "engineers"]
    assert agent.tasks == [VALIDATED_TASK]


@pytest.mark.parametrize("agent_type", [TypedEventsAgent, TypedAsyncStreamAgent])
def test_aevents_validates_input_exactly_once(agent_type):
    agent = agent_type()
    tactic = CountingPydanticAITactic(agent, input_type=BriefInput)

    events = asyncio.run(collect(tactic.aevents({"topic": "x"})))

    assert [event.data for event in events] == ["x", "engineers"]
    assert agent.tasks == [VALIDATED_TASK]
    assert tactic.validations == 1


def test_stream_rejects_invalid_input_before_calling_agent():
    agent = TypedStreamAgent()
    tactic = PydanticAITactic(agent, input_type=BriefInput)

    with pytest.raises(SchemaError):
        tactic.stream({"topic": 123})
    with pytest.raises(SchemaError):
        asyncio.run(collect(tactic.aevents({"audience": "ops"})))
    assert agent.tasks == []


def test_service_stream_route_validates_typed_pydantic_ai_input():
    agent = TypedStreamAgent()
    app = create_tactic_app(PydanticAITactic(agent, input_type=BriefInput))

    status, text = request(app, "POST", "/stream", json={"input": {"topic": "x"}})

    assert status == 200
    payloads = sse_payloads(text)
    assert [payload["kind"] for payload in payloads] == ["message", "message"]
    assert [payload["data"] for payload in payloads] == ["x", "engineers"]
    assert agent.tasks == [VALIDATED_TASK]


# -- P3: aevents yields JSON-safe TacticEvents -------------------------------


@dataclasses.dataclass
class FakeTextPart:
    content: str
    timestamp: datetime
    part_kind: str = "text"


@dataclasses.dataclass
class FakePartStartEvent:
    index: int
    part: FakeTextPart
    event_kind: str = "part_start"


class FakeOutput(BaseModel):
    title: str


@dataclasses.dataclass
class FakeRunResult:
    output: FakeOutput
    _state: Any = dataclasses.field(default_factory=lambda: {"messages": ["private"]}, repr=False)


@dataclasses.dataclass
class FakeRunResultEvent:
    result: FakeRunResult
    event_kind: str = "agent_run_result"


class FakeStatusEvent(BaseModel):
    detail: str
    at: datetime
    event_kind: str = "status"


@dataclasses.dataclass
class FakeBinaryEvent:
    data: bytes
    labels: tuple[str, ...]
    event_kind: str = "binary"


class OpaqueEvent:
    event_kind = "not a token/kind"

    def __str__(self):
        return "opaque-event"


STAMP = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)


class FakeNativeEventsAgent:
    name = "native-events"
    output_type = FakeOutput

    def run_stream_events(self, task, **kwargs):
        return EventsHandle(
            [
                FakePartStartEvent(index=0, part=FakeTextPart(content=task, timestamp=STAMP)),
                FakeStatusEvent(detail="working", at=STAMP),
                FakeBinaryEvent(data=b"\x89PNG", labels=("image", "png")),
                OpaqueEvent(),
                TacticEvent(kind="custom", data={"already": "wrapped"}),
                FakeRunResultEvent(result=FakeRunResult(output=FakeOutput(title="done"))),
            ]
        )


def test_aevents_wraps_runtime_events_in_json_safe_tactic_events():
    tactic = PydanticAITactic(FakeNativeEventsAgent(), input_type=str)

    events = asyncio.run(collect(tactic.aevents("hello")))

    assert all(isinstance(event, TacticEvent) for event in events)
    assert [event.kind for event in events] == [
        "part_start",
        "status",
        "binary",
        "message",
        "custom",
        "agent_run_result",
    ]
    assert [event.metadata for event in events] == [
        {"event_type": "FakePartStartEvent"},
        {"event_type": "FakeStatusEvent"},
        {"event_type": "FakeBinaryEvent"},
        {"event_type": "OpaqueEvent"},
        {},
        {"event_type": "FakeRunResultEvent"},
    ]
    assert events[0].data == {
        "index": 0,
        "part": {"content": "hello", "timestamp": "2026-01-02T03:04:05Z", "part_kind": "text"},
        "event_kind": "part_start",
    }
    assert events[1].data == {
        "detail": "working",
        "at": "2026-01-02T03:04:05Z",
        "event_kind": "status",
    }
    assert events[2].data == {
        "data": "iVBORw==",
        "labels": ["image", "png"],
        "event_kind": "binary",
    }
    assert events[3].data == "opaque-event"
    assert events[4].data == {"already": "wrapped"}
    assert events[5].data == {
        "result": {"output": {"title": "done"}},
        "event_kind": "agent_run_result",
    }
    assert json.loads(json.dumps([event.data for event in events])) == [
        event.data for event in events
    ]


def test_service_stream_route_sends_json_for_runtime_events():
    app = create_tactic_app(PydanticAITactic(FakeNativeEventsAgent(), input_type=str))

    status, text = request(app, "POST", "/stream", json={"input": "hello"})

    assert status == 200
    payloads = sse_payloads(text)
    assert all(isinstance(payload, dict) for payload in payloads)
    assert payloads[0]["kind"] == "part_start"
    assert payloads[0]["data"]["part"]["content"] == "hello"
    assert payloads[-1]["kind"] == "agent_run_result"
    assert payloads[-1]["data"] == {
        "result": {"output": {"title": "done"}},
        "event_kind": "agent_run_result",
    }
    assert "private" not in text
    assert "FakePartStartEvent(" not in text


def test_aevents_with_real_pydantic_ai_agent_yields_json_safe_tactic_events():
    Agent, TestModel = real_pydantic_ai()
    tactic = PydanticAITactic(
        Agent(TestModel(custom_output_text="done"), name="events"),
        input_type=str,
    )

    events = asyncio.run(collect(tactic.aevents("hello")))

    assert all(isinstance(event, TacticEvent) for event in events)
    assert events[0].kind == "part_start"
    assert events[0].metadata == {"event_type": "PartStartEvent"}
    assert events[-1].kind == "agent_run_result"
    assert events[-1].data == {"result": {"output": "done"}, "event_kind": "agent_run_result"}
    json.dumps([event.data for event in events])


# -- P4: descriptions never come from Pydantic AI's own docstrings ----------


class DocumentedTactic(PydanticAITactic):
    """Write a structured brief."""


class AttributeTactic(PydanticAITactic):
    description = "Answer support questions."


class InheritedDocTactic(DocumentedTactic):
    pass


class PlainAgent:
    name = "plain"
    output_type = str

    def run_sync(self, task, **kwargs):
        return task


class DocumentedAgent(PlainAgent):
    """User agent docstring."""


class RuntimeOwnedAgent(PlainAgent):
    """Runtime class documentation that is not a tactic description."""


RuntimeOwnedAgent.__module__ = "pydantic_ai.agent"


def test_description_ignores_runtime_and_adapter_class_docstrings():
    assert PydanticAITactic(RuntimeOwnedAgent()).info().description == ""
    assert PydanticAITactic(PlainAgent()).info().description == ""


def test_description_uses_user_provided_sources():
    assert (
        PydanticAITactic(RuntimeOwnedAgent(), description="Explicit.").info().description
        == "Explicit."
    )
    assert (
        PydanticAITactic(RuntimeOwnedAgent(), config={"description": "Configured."})
        .info()
        .description
        == "Configured."
    )
    assert DocumentedTactic(RuntimeOwnedAgent()).info().description == "Write a structured brief."
    assert InheritedDocTactic(PlainAgent()).info().description == "Write a structured brief."
    assert AttributeTactic(PlainAgent()).info().description == "Answer support questions."
    assert PydanticAITactic(DocumentedAgent()).info().description == "User agent docstring."
    assert DocumentedTactic(PlainAgent(), description="").info().description == ""


def test_description_with_real_pydantic_ai_agents():
    Agent, TestModel = real_pydantic_ai()
    from pydantic_ai.agent import WrapperAgent

    class SupportAgent(WrapperAgent):
        """Answer support questions."""

    agent = Agent(TestModel(), name="brief")

    assert PydanticAITactic(agent).info().description == ""
    assert PydanticAITactic(WrapperAgent(agent)).info().description == ""
    assert PydanticAITactic(SupportAgent(agent)).info().description == "Answer support questions."
    described = Agent(TestModel(), name="described", description="Write briefs.")
    assert PydanticAITactic(described).info().description == "Write briefs."
    status, text = request(create_tactic_app(PydanticAITactic(agent)), "GET", "/info")
    assert status == 200
    assert json.loads(text)["description"] == ""


# -- P5: result mode defaults the output type to Any -------------------------


class RawResult:
    def __init__(self, output):
        self.output = output


class RawResultAgent:
    name = "raw-result"
    output_type = str

    def run_sync(self, task, **kwargs):
        return RawResult(f"{task}!")


@pytest.mark.parametrize(
    "kwargs",
    [{"result_mode": "result"}, {"config": {"result_mode": "result"}}],
)
def test_result_mode_returns_raw_result_without_output_type(kwargs):
    tactic = PydanticAITactic(RawResultAgent(), **kwargs)

    result = tactic.run("hi")
    async_result = asyncio.run(tactic.arun("hi"))

    assert tactic.output_type is Any
    assert isinstance(result, RawResult) and result.output == "hi!"
    assert isinstance(async_result, RawResult) and async_result.output == "hi!"


def test_result_mode_keeps_explicit_output_type_validation():
    tactic = PydanticAITactic(RawResultAgent(), result_mode="result", output_type=str)

    with pytest.raises(SchemaError):
        tactic.run("hi")


def test_output_mode_still_defaults_to_agent_output_type():
    tactic = PydanticAITactic(RawResultAgent())

    assert tactic.output_type is str
    assert tactic.run("hi") == "hi!"


def test_result_mode_with_real_pydantic_ai_agent():
    Agent, TestModel = real_pydantic_ai()
    tactic = PydanticAITactic(
        Agent(TestModel(custom_output_text="done"), name="raw"),
        result_mode="result",
    )

    result = tactic.run("hello")

    assert tactic.output_type is Any
    assert result.output == "done"
    assert asyncio.run(tactic.arun("hello")).output == "done"
