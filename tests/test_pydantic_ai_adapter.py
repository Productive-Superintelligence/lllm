import asyncio

from lllm import CallContext
from lllm.runtimes import PydanticAITactic, tactic_as_tool


class Result:
    def __init__(self, output):
        self.output = output


class FakeAgent:
    name = "fake"
    output_type = str

    def __init__(self):
        self.seen_kwargs = None

    def run_sync(self, task, **kwargs):
        self.seen_kwargs = kwargs
        return Result(f"{task}:{kwargs.get('suffix', '')}")

    def output_json_schema(self):
        return {"type": "string", "title": "FakeOutput"}


class AsyncStream:
    def __init__(self, values):
        self.values = values

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        for value in self.values:
            yield value


class FakeStreamAgent:
    def run_stream(self, task, **kwargs):
        return AsyncStream([task, kwargs["suffix"]])


async def collect(iterator):
    return [item async for item in iterator]


def test_pydantic_ai_adapter_maps_result_output_and_context_metadata():
    agent = FakeAgent()
    tactic = PydanticAITactic.from_agent(
        agent,
        input_type=str,
        run_kwargs={"suffix": "ok"},
    )

    result = tactic.run(
        "hello",
        context=CallContext(
            trace_id="trace-1",
            service_ref="psi://demo/pkg/services/api",
            endpoint="run",
            metadata={"caller": "test"},
        ),
    )

    assert result == "hello:ok"
    assert agent.seen_kwargs["metadata"]["caller"] == "test"
    assert agent.seen_kwargs["metadata"]["lllm_trace_id"] == "trace-1"
    assert agent.seen_kwargs["metadata"]["lllm_service_ref"] == "psi://demo/pkg/services/api"
    assert agent.seen_kwargs["metadata"]["lllm_endpoint"] == "run"
    assert tactic.info().output_schema == {"type": "string", "title": "FakeOutput"}
    assert tactic.capabilities() == {"run", "arun"}


def test_pydantic_ai_adapter_supports_async_streams():
    tactic = PydanticAITactic.from_agent(
        FakeStreamAgent(),
        input_type=str,
        output_type=str,
        run_kwargs={"suffix": "done"},
    )

    assert tactic.supports("stream")
    assert asyncio.run(collect(tactic.astream("hello"))) == ["hello", "done"]


def test_tactic_as_tool_is_plain_callable():
    tactic = PydanticAITactic.from_agent(FakeAgent(), input_type=str)
    tool = tactic_as_tool(tactic, name="answer tool")

    assert tool.__name__ == "answer_tool"
    assert tool("hi") == "hi:"
