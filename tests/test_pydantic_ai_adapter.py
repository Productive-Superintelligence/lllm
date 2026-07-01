import asyncio
from types import MappingProxyType

import pytest
from pydantic import BaseModel, ValidationError

from lllm import CallContext, Tactic
from lllm.runtimes import PydanticAITactic, PydanticAITacticConfig, tactic_as_tool


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


class FeatureAgent:
    name = "features"
    output_type = dict

    def __init__(self):
        self.seen_kwargs = {}

    def run_sync(self, task, *, metadata=None, **kwargs):
        self.seen_kwargs = {"metadata": metadata or {}, **kwargs}
        return Result(
            {
                "task": task,
                "durable_run_id": kwargs.get("durable_run_id"),
                "graph_node": kwargs.get("graph_node"),
                "temperature": kwargs.get("model_settings", {}).get("temperature"),
            }
        )


class MutatingKwargsAgent:
    name = "mutating-kwargs"
    output_type = str

    def __init__(self):
        self.snapshots = []

    def run_sync(self, task, **kwargs):
        self.snapshots.append(
            {
                "temperature": kwargs["model_settings"]["temperature"],
                "flags": list(kwargs["runtime_flags"]),
                "handle": kwargs["deps"]["handle"],
            }
        )
        kwargs["model_settings"]["temperature"] = 99
        kwargs["runtime_flags"].append("mutated")
        kwargs["deps"]["agent_mutated"] = True
        return Result(task)


async def collect(iterator):
    return [item async for item in iterator]


def test_pydantic_ai_adapter_maps_result_output_and_context_metadata():
    agent = FakeAgent()
    example = {"input": "hello", "output": "hello:ok"}
    tactic = PydanticAITactic.from_agent(
        agent,
        input_type=str,
        run_kwargs={"suffix": "ok"},
        description="Fake package tactic.",
        package_ref="psi://demo/fake/tactics/fake",
        service_ref="psi://demo/fake/services/api",
        examples=[example],
        metadata={"owner": "tests"},
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
    info = tactic.info()
    assert info.description == "Fake package tactic."
    assert info.package_ref == "psi://demo/fake/tactics/fake"
    assert info.service_ref == "psi://demo/fake/services/api"
    assert info.examples == [example]
    assert info.metadata == {"owner": "tests"}
    assert info.output_schema == {"type": "string", "title": "FakeOutput"}
    assert tactic.capabilities() == {"run", "arun"}


@pytest.mark.parametrize("name", ["", "   "])
def test_pydantic_ai_adapter_rejects_explicit_blank_names(name):
    with pytest.raises(ValueError, match="name"):
        PydanticAITactic.from_agent(FakeAgent(), name=name)


def test_pydantic_ai_adapter_isolates_forwarded_context_metadata():
    agent = FakeAgent()
    tactic = PydanticAITactic.from_agent(agent, input_type=str)
    context = CallContext(metadata={"nested": {"value": 1}})

    tactic.run("hello", context=context)
    agent.seen_kwargs["metadata"]["nested"]["value"] = 2

    assert context.metadata == {"nested": {"value": 1}}


def test_pydantic_ai_adapter_accepts_package_metadata_from_config():
    tactic = PydanticAITactic.from_agent(
        FakeAgent(),
        config={
            "input_type": str,
            "description": "Configured package tactic.",
            "package_ref": "psi://demo/configured/tactics/fake",
            "service_ref": "psi://demo/configured/services/api",
            "examples": [{"input": "hi", "output": "hi:"}],
            "metadata": {"source": "config"},
        },
    )

    info = tactic.info()

    assert info.description == "Configured package tactic."
    assert info.package_ref == "psi://demo/configured/tactics/fake"
    assert info.service_ref == "psi://demo/configured/services/api"
    assert info.examples == [{"input": "hi", "output": "hi:"}]
    assert info.metadata == {"source": "config"}


def test_pydantic_ai_adapter_info_does_not_evaluate_optional_properties():
    class OptionalPropertyAgent(FakeAgent):
        @property
        def run_stream(self):
            raise AssertionError("run_stream property should not be evaluated")

        @property
        def run_stream_sync(self):
            raise AssertionError("run_stream_sync property should not be evaluated")

        @property
        def run_stream_events(self):
            raise AssertionError("run_stream_events property should not be evaluated")

        @property
        def output_json_schema(self):
            raise AssertionError("schema property should not be evaluated")

    tactic = PydanticAITactic.from_agent(OptionalPropertyAgent(), input_type=str)
    info = tactic.info()

    assert tactic.capabilities() == {"run", "arun"}
    assert info.capabilities == ("arun", "run")
    assert info.output_schema == {"type": "string"}


@pytest.mark.parametrize("config", [[], "", 0, False])
def test_pydantic_ai_adapter_rejects_falsey_non_mapping_config(config):
    with pytest.raises(ValidationError):
        PydanticAITactic.from_agent(FakeAgent(), config=config)


def test_pydantic_ai_config_isolates_mutable_inputs():
    provider_settings = {"timeout": 30}
    run_kwargs = {
        "model_settings": {"temperature": 0},
        "provider_settings": MappingProxyType(provider_settings),
    }
    examples = [{"input": "hi", "output": "hi:"}]
    metadata = {"labels": ["config"]}
    config = PydanticAITacticConfig(
        run_kwargs=run_kwargs,
        examples=examples,
        metadata=metadata,
    )

    run_kwargs["model_settings"]["temperature"] = 7
    provider_settings["timeout"] = 60
    examples[0]["input"] = "changed"
    metadata["labels"].append("changed")

    assert config.run_kwargs == {
        "model_settings": {"temperature": 0},
        "provider_settings": {"timeout": 30},
    }
    assert config.examples == [{"input": "hi", "output": "hi:"}]
    assert config.metadata == {"labels": ["config"]}


@pytest.mark.parametrize("field_name", ["description", "package_ref", "service_ref"])
def test_pydantic_ai_config_rejects_bytes_for_package_metadata(field_name):
    with pytest.raises(ValidationError) as exc_info:
        PydanticAITacticConfig.model_validate({field_name: b"not-text"})

    assert exc_info.value.errors()[0]["type"] == "string_type"


@pytest.mark.parametrize(
    "package_ref",
    [
        "not-a-ref",
        "psi://demo/echo/channels/events",
        "psi://demo/echo/tactics/echo?token=raw",
        "psi://demo/echo/tactics/bad name",
        "psi://demo/echo/tactics/echo;name",
    ],
)
def test_pydantic_ai_config_rejects_malformed_package_refs(package_ref):
    with pytest.raises(ValidationError, match="package_ref"):
        PydanticAITacticConfig(package_ref=package_ref)


@pytest.mark.parametrize(
    "service_ref",
    [
        "not-a-ref",
        "ftp://service",
        "http://user:pass@service",
        "http://service?env=dev",
        "http://service/base//api",
        "http://service/base%2Fapi",
        "http://service/base\\api",
        "http://service/base:api",
        "psi://demo/echo/tactics/echo",
        "psi://demo/echo/services/api?token=raw",
        "psi://demo/echo/services/bad name",
        "psi://demo/echo/services/api;name",
    ],
)
def test_pydantic_ai_config_rejects_malformed_service_refs(service_ref):
    with pytest.raises(ValidationError, match="service_ref"):
        PydanticAITacticConfig(service_ref=service_ref)


def test_pydantic_ai_config_rejects_non_string_metadata_keys():
    with pytest.raises(ValidationError, match="metadata"):
        PydanticAITacticConfig(metadata={"headers": {b"x-policy": "demo"}})


def test_pydantic_ai_adapter_rejects_non_string_metadata_keys():
    with pytest.raises(TypeError, match="metadata"):
        PydanticAITactic.from_agent(
            FakeAgent(),
            metadata={b"owner": "tests"},  # type: ignore[arg-type]
        )


def test_pydantic_ai_config_rejects_non_string_example_keys():
    with pytest.raises(ValidationError, match="examples"):
        PydanticAITacticConfig(examples=[{"input": {b"text": "hi"}}])


def test_pydantic_ai_adapter_rejects_non_string_example_keys():
    with pytest.raises(TypeError, match="examples"):
        PydanticAITactic.from_agent(
            FakeAgent(),
            examples=[{b"input": "hi"}],  # type: ignore[list-item]
        )


@pytest.mark.parametrize("value", ["false", "true", 0, 1])
def test_pydantic_ai_config_rejects_non_bool_context_metadata_flag(value):
    with pytest.raises(ValidationError):
        PydanticAITacticConfig.model_validate({"include_context_metadata": value})


@pytest.mark.parametrize("examples", [(), ({"input": "hi"},), "", False, 0])
def test_pydantic_ai_config_rejects_non_list_examples(examples):
    with pytest.raises(ValidationError):
        PydanticAITacticConfig.model_validate({"examples": examples})


@pytest.mark.parametrize("value", ["false", "true", 0, 1])
def test_pydantic_ai_adapter_rejects_non_bool_context_metadata_override(value):
    with pytest.raises(TypeError, match="include_context_metadata"):
        PydanticAITactic.from_agent(
            FakeAgent(),
            include_context_metadata=value,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("run_kwargs", [[], [("suffix", "ok")], "bad", 123])
def test_pydantic_ai_adapter_rejects_non_mapping_run_kwargs(run_kwargs):
    with pytest.raises(TypeError, match="run_kwargs"):
        PydanticAITactic.from_agent(
            FakeAgent(),
            input_type=str,
            run_kwargs=run_kwargs,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("run_kwargs", [{b"suffix": "ok"}, {1: "ok"}])
def test_pydantic_ai_adapter_rejects_non_string_run_kwarg_keys(run_kwargs):
    with pytest.raises(TypeError, match="run_kwargs"):
        PydanticAITactic.from_agent(
            FakeAgent(),
            input_type=str,
            run_kwargs=run_kwargs,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("run_kwargs", [{b"suffix": "ok"}, {1: "ok"}])
def test_pydantic_ai_config_rejects_non_string_run_kwarg_keys(run_kwargs):
    with pytest.raises(ValidationError, match="run_kwargs"):
        PydanticAITacticConfig(run_kwargs=run_kwargs)  # type: ignore[arg-type]


def test_pydantic_ai_adapter_supports_async_streams():
    tactic = PydanticAITactic.from_agent(
        FakeStreamAgent(),
        input_type=str,
        output_type=str,
        run_kwargs={"suffix": "done"},
    )

    assert tactic.supports("stream")
    assert asyncio.run(collect(tactic.astream("hello"))) == ["hello", "done"]


def test_pydantic_ai_adapter_preserves_runtime_owned_kwargs():
    agent = FeatureAgent()
    tactic = PydanticAITactic.from_agent(
        agent,
        input_type=str,
        output_type=dict,
        run_kwargs={
            "model_settings": {"temperature": 0},
            "deps": {"db": "fake"},
            "eval_hook": "offline",
            "tool_approval": "runtime-owned",
        },
    )

    output = tactic.run(
        "hello",
        context=CallContext(trace_id="trace-2", metadata={"caller": "test"}),
        durable_run_id="durable-1",
        graph_node="workflow.step",
    )

    assert output == {
        "task": "hello",
        "durable_run_id": "durable-1",
        "graph_node": "workflow.step",
        "temperature": 0,
    }
    assert agent.seen_kwargs["metadata"]["lllm_trace_id"] == "trace-2"
    assert agent.seen_kwargs["metadata"]["caller"] == "test"
    assert agent.seen_kwargs["deps"] == {"db": "fake"}
    assert agent.seen_kwargs["eval_hook"] == "offline"
    assert agent.seen_kwargs["tool_approval"] == "runtime-owned"


def test_pydantic_ai_adapter_isolates_mutable_run_kwargs_containers():
    agent = MutatingKwargsAgent()
    handle = object()
    run_kwargs = {
        "model_settings": {"temperature": 0},
        "runtime_flags": ["initial"],
        "deps": {"handle": handle},
    }
    tactic = PydanticAITactic.from_agent(agent, input_type=str, run_kwargs=run_kwargs)

    run_kwargs["model_settings"]["temperature"] = 7
    run_kwargs["runtime_flags"].append("caller")
    run_kwargs["deps"]["caller_mutated"] = True

    assert tactic.run("first") == "first"
    assert tactic.run("second") == "second"

    assert agent.snapshots == [
        {"temperature": 0, "flags": ["initial"], "handle": handle},
        {"temperature": 0, "flags": ["initial"], "handle": handle},
    ]
    assert tactic.run_kwargs == {
        "model_settings": {"temperature": 0},
        "runtime_flags": ["initial"],
        "deps": {"handle": handle},
    }


def test_pydantic_ai_adapter_isolates_mutable_call_kwargs_containers():
    agent = MutatingKwargsAgent()
    handle = object()
    tactic = PydanticAITactic.from_agent(agent, input_type=str)
    model_settings = {"temperature": 0}
    runtime_flags = ["call"]
    deps = {"handle": handle}

    assert (
        tactic.run(
            "hello",
            model_settings=model_settings,
            runtime_flags=runtime_flags,
            deps=deps,
        )
        == "hello"
    )

    assert model_settings == {"temperature": 0}
    assert runtime_flags == ["call"]
    assert deps == {"handle": handle}
    assert agent.snapshots == [{"temperature": 0, "flags": ["call"], "handle": handle}]


def test_pydantic_ai_adapter_does_not_override_user_metadata():
    agent = FakeAgent()
    tactic = PydanticAITactic.from_agent(agent, input_type=str)

    tactic.run(
        "hello",
        context=CallContext(trace_id="trace-owned", metadata={"caller": "lllm"}),
        metadata={"runtime": "owned"},
    )

    assert agent.seen_kwargs["metadata"] == {"runtime": "owned"}


def test_tactic_as_tool_is_plain_callable():
    tactic = PydanticAITactic.from_agent(FakeAgent(), input_type=str)
    tool = tactic_as_tool(tactic, name="answer tool")

    assert tool.__name__ == "answer_tool"
    assert tool("hi") == "hi:"


@pytest.mark.parametrize("name", ["", "   ", " answer ", 123])
def test_tactic_as_tool_rejects_malformed_explicit_names(name):
    tactic = PydanticAITactic.from_agent(FakeAgent(), input_type=str)

    with pytest.raises(ValueError, match="tool name must be non-empty"):
        tactic_as_tool(tactic, name=name)  # type: ignore[arg-type]


def test_tactic_as_tool_infers_name_only_when_name_is_none():
    tactic = PydanticAITactic.from_agent(FakeAgent(), input_type=str, name="answer tool")
    tool = tactic_as_tool(tactic, name=None)

    assert tool.__name__ == "answer_tool"


class AddInput(BaseModel):
    left: int
    right: int


class AddTactic(Tactic[AddInput, int]):
    name = "add numbers"
    input_type = AddInput
    output_type = int

    def _run(self, input_value, *, context=None):
        return input_value.left + input_value.right


def test_tactic_as_tool_supports_kwargs_for_pydantic_inputs():
    tool = tactic_as_tool(AddTactic(), parameter_mode="kwargs")

    assert tool.__name__ == "add_numbers"
    assert tool(left=2, right=3) == 5
    assert "left" in tool.__signature__.parameters
