import importlib.util
import re
from pathlib import Path

from fastapi.testclient import TestClient

from lllm import CallContext
from lllm.services import create_tactic_app


ROOT = Path(__file__).resolve().parents[1]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_documented_example_paths_exist():
    pattern = re.compile(r"`(examples/[^`]+)`")
    text_paths = [ROOT / "README.md"]
    text_paths.extend((ROOT / "docs").rglob("*.md"))

    missing = []
    for path in sorted(text_paths):
        for match in pattern.finditer(path.read_text(encoding="utf-8")):
            example_path = ROOT / match.group(1)
            if not example_path.exists():
                missing.append(f"{path.relative_to(ROOT)} -> {match.group(1)}")

    assert missing == []


def test_echo_service_example_builds_tactic():
    module = load_module(ROOT / "examples" / "echo_service" / "tactics.py", "echo_tactics")
    tactic = module.build_tactic()

    assert tactic.run({"text": "hello"}).text == "HELLO"


def test_pydantic_ai_fake_agent_example_builds_tactic():
    module = load_module(
        ROOT / "examples" / "pydantic_ai_tactic" / "fake_agent.py",
        "fake_agent",
    )
    tactic = module.build_tactic()

    assert tactic.run("hello") == "HELLO"


def test_native_dialog_example_builds_transcript_and_retry_branch():
    module = load_module(
        ROOT / "examples" / "native_dialog" / "demo.py",
        "native_dialog_demo",
    )

    dialog = module.build_dialog()
    retry = module.build_retry_dialog()

    assert dialog.owner == "planner"
    assert dialog.head.content == "You are a careful planning assistant."
    assert dialog.tail.name == "label"
    assert dialog.tail.metadata["function_call"]["result"] == "Native Core"
    assert retry.parent is not None
    assert retry.depth == 1


def test_native_service_example_runs_locally_and_over_http():
    module = load_module(
        ROOT / "examples" / "native_service" / "tactics.py",
        "native_service_tactics",
    )
    tactic = module.build_tactic()

    output = tactic.run(
        {"topic": "native services", "audience": "operators"},
        context=CallContext(trace_id="trace-native"),
    )

    assert output.title == "Native Services Brief"
    assert output.trace_id == "trace-native"
    assert output.transcript[0].role == "system"
    assert output.transcript[-1].name == "native-brief"
    assert tactic.info().runtime_kind == "native"
    assert tactic.info().metadata == {"example": "native-service"}

    client = TestClient(create_tactic_app(tactic))
    response = client.post(
        "/run",
        json={
            "input": {"topic": "native services", "audience": "operators"},
            "context": {"trace_id": "trace-http"},
        },
    )

    assert response.status_code == 200
    assert response.json()["output"]["trace_id"] == "trace-http"
    assert response.json()["output"]["transcript"][0]["role"] == "system"


def test_structured_pydantic_ai_example_runs_streams_and_builds_tool():
    module = load_module(
        ROOT / "examples" / "pydantic_ai_tactic" / "structured_agent.py",
        "structured_agent",
    )
    agent = module.FakeStructuredAgent()
    tactic = module.build_tactic(agent)

    output = tactic.run(
        {"topic": "package refs", "audience": "maintainers"},
        context=CallContext(trace_id="trace-structured"),
    )

    assert output.title == "Package Refs for maintainers"
    assert output.trace_id == "trace-structured"
    assert agent.last_task == {"topic": "package refs", "audience": "maintainers"}
    assert tactic.info().output_schema["title"] == "BriefOutput"
    assert list(
        tactic.stream(module.BriefInput(topic="channels", audience="operators"))
    ) == ["topic:channels", "audience:operators"]

    tool = module.build_tool()
    tool_output = tool(topic="tactics", audience="runtime users")

    assert tool_output.title == "Tactics for runtime users"


def test_pydantic_ai_surrounding_features_example_preserves_runtime_ownership():
    module = load_module(
        ROOT / "examples" / "pydantic_ai_tactic" / "surrounding_features.py",
        "surrounding_features",
    )

    output, agent = module.run_demo()

    assert output["model"] == "fake-provider:small"
    assert output["instrumentation"] == ["logfire", "opentelemetry"]
    assert output["trace_id"] == "trace-surrounding"
    assert output["durable_run_id"] == "durable-1"
    assert output["graph_node"] == "plan.step"
    assert agent.seen["instrumented"] is True
    assert agent.seen["instrumentation"] == ("logfire", "opentelemetry")
    assert agent.seen["kwargs"]["model_settings"] == {"temperature": 0}
    assert agent.seen["kwargs"]["deps"] == {"vector_store": "fake"}
    assert agent.seen["kwargs"]["eval_hook"] == "offline-score"
    assert agent.seen["kwargs"]["tool_approval"] == "runtime-owned"
