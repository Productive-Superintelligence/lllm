import ast
import os
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError
from typing_extensions import TypedDict

from lllm import (
    CallContext,
    CallResult,
    CallTrace,
    SchemaError,
    Tactic,
    TacticEvent,
    TacticInfo,
    as_tactic,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ROOT = ROOT / "lllm" / "protocol"
FORBIDDEN_PROTOCOL_IMPORT_PREFIXES = (
    "fastapi",
    "lllm.cli",
    "lllm.create",
    "lllm.integrations",
    "lllm.runtimes",
    "lllm.services",
    "pydantic_ai",
    "psihub",
    "sssn",
)


class EchoInput(BaseModel):
    text: str


class EchoOutput(BaseModel):
    text: str


class EchoTactic(Tactic[EchoInput, EchoOutput]):
    name = "echo"
    input_type = EchoInput
    output_type = EchoOutput

    def _run(self, input_value, *, context=None):
        suffix = ""
        if context is not None:
            suffix = context.metadata.get("suffix", "")
        return {"text": input_value.text.upper() + suffix}


@dataclass(frozen=True)
class BatchInput:
    text: str
    repeat: int


class BatchOutput(TypedDict):
    tokens: list[str]
    count: int


class BatchTactic(Tactic[BatchInput, BatchOutput]):
    name = "batch"
    input_type = BatchInput
    output_type = BatchOutput

    def _run(self, input_value, *, context=None):
        assert isinstance(input_value, BatchInput)
        tokens = [input_value.text] * input_value.repeat
        return {"tokens": tokens, "count": len(tokens)}


def test_call_context_isolates_mutable_metadata_inputs():
    metadata = {"nested": {"value": 1}}
    tags = {"kind": "demo"}
    context = CallContext(metadata=metadata, tags=tags)

    metadata["nested"]["value"] = 2
    tags["kind"] = "changed"

    assert context.metadata == {"nested": {"value": 1}}
    assert context.tags == {"kind": "demo"}


def test_tactic_event_isolates_mutable_payload_inputs():
    data = {"chunks": ["hello"]}
    metadata = {"labels": ["stream"]}
    event = TacticEvent(kind="progress", data=data, metadata=metadata)

    data["chunks"].append("changed")
    metadata["labels"].append("changed")

    assert event.data == {"chunks": ["hello"]}
    assert event.metadata == {"labels": ["stream"]}

    tags = ["result"]
    result = TacticEvent.result({"items": [1]}, tags=tags)
    tags.append("changed")

    assert result.data == {"items": [1]}
    assert result.metadata == {"tags": ["result"]}


def test_protocol_info_and_trace_models_isolate_mutable_inputs():
    input_schema = {"properties": {"text": {"type": "string"}}}
    output_schema = {"properties": {"text": {"type": "string"}}}
    examples = [{"input": {"text": "hello"}}]
    metadata = {"labels": ["info"]}
    info = TacticInfo(
        name="echo",
        input_schema=input_schema,
        output_schema=output_schema,
        examples=examples,
        metadata=metadata,
    )
    trace_metadata = {"labels": ["trace"]}
    trace = CallTrace(request_id="req", tactic="echo", metadata=trace_metadata)

    input_schema["properties"]["text"]["type"] = "integer"
    output_schema["properties"]["text"]["type"] = "integer"
    examples[0]["input"]["text"] = "changed"
    metadata["labels"].append("changed")
    trace_metadata["labels"].append("changed")

    assert info.input_schema == {"properties": {"text": {"type": "string"}}}
    assert info.output_schema == {"properties": {"text": {"type": "string"}}}
    assert info.examples == [{"input": {"text": "hello"}}]
    assert info.metadata == {"labels": ["info"]}
    assert trace.metadata == {"labels": ["trace"]}


@pytest.mark.parametrize(
    "factory",
    [
        lambda: CallContext(request_id=b"req"),
        lambda: CallContext(caller=b"caller"),
        lambda: CallContext(tags={b"kind": "demo"}),
        lambda: CallContext(tags={"kind": b"demo"}),
        lambda: TacticEvent(id=b"event"),
        lambda: TacticEvent(kind=b"progress"),
        lambda: TacticInfo(name=b"echo"),
        lambda: TacticInfo(name="echo", capabilities=(b"run",)),
        lambda: CallTrace(request_id=b"req", tactic="echo"),
        lambda: CallTrace(request_id="req", tactic=b"echo"),
    ],
)
def test_protocol_string_fields_reject_bytes(factory):
    with pytest.raises(ValidationError):
        factory()


def test_call_result_isolates_mutable_output_and_trace():
    output = {"items": [1]}
    trace = CallTrace(request_id="req", tactic="echo", metadata={"labels": ["trace"]})
    result = CallResult(output=output, trace=trace)

    output["items"].append(2)
    trace.metadata["labels"].append("changed")
    result.trace.metadata["labels"].append("result")

    assert result.output == {"items": [1]}
    assert result.trace.metadata == {"labels": ["trace", "result"]}
    assert trace.metadata == {"labels": ["trace", "changed"]}


def test_tactic_validates_and_returns_trace():
    result = EchoTactic().run(
        {"text": "hello"},
        context=CallContext(metadata={"suffix": "!"}),
        return_trace=True,
    )

    assert result.output == EchoOutput(text="HELLO!")
    assert result.trace.state == "success"
    assert result.trace.tactic == "echo"
    assert result.trace.latency_ms is not None


def test_tactic_info_exports_json_schema():
    info = EchoTactic().info()

    assert info.name == "echo"
    assert info.runtime_kind == "python"
    assert info.input_schema["properties"]["text"]["type"] == "string"
    assert info.output_schema["properties"]["text"]["type"] == "string"


def test_tactic_info_and_trace_metadata_are_isolated():
    example = {"input": {"text": "hello"}}
    metadata = {"labels": ["demo"]}
    tactic = EchoTactic(examples=[example], metadata=metadata)

    example["input"]["text"] = "changed"
    metadata["labels"].append("changed")
    info = tactic.info()
    assert info.examples == [{"input": {"text": "hello"}}]
    assert info.metadata == {"labels": ["demo"]}

    info.examples[0]["input"]["text"] = "mutated"
    info.metadata["labels"].append("mutated")
    assert tactic.info().examples == [{"input": {"text": "hello"}}]
    assert tactic.info().metadata == {"labels": ["demo"]}

    context = CallContext(metadata={"labels": ["context"]})
    result = tactic.run({"text": "hello"}, context=context, return_trace=True)
    context.metadata["labels"].append("changed")
    result.trace.metadata["labels"].append("mutated")

    next_result = tactic.run({"text": "hello"}, context=context, return_trace=True)
    assert result.trace.metadata == {"labels": ["context", "mutated"]}
    assert next_result.trace.metadata == {"labels": ["context", "changed"]}


def test_tactic_rejects_invalid_input():
    with pytest.raises(SchemaError):
        EchoTactic().run({"text": 123})


def test_plain_callable_wraps_as_tactic_with_annotations():
    def shout(text: str, *, context=None) -> str:
        assert context.metadata["caller"] == "test"
        return text.upper()

    example = {"input": "hello", "output": "HELLO"}
    tactic = as_tactic(
        shout,
        description="Uppercase text.",
        package_ref="psi://demo/echo/tactics/shout",
        service_ref="psi://demo/echo/services/api",
        examples=[example],
        metadata={"owner": "tests"},
    )
    info = tactic.info()

    assert info.description == "Uppercase text."
    assert info.package_ref == "psi://demo/echo/tactics/shout"
    assert info.service_ref == "psi://demo/echo/services/api"
    assert info.examples == [example]
    assert info.metadata == {"owner": "tests"}
    assert info.input_schema["type"] == "string"
    assert tactic.run("hello", context=CallContext(metadata={"caller": "test"})) == "HELLO"


def test_tactic_accepts_dataclass_input_and_typed_dict_output():
    tactic = BatchTactic()
    info = tactic.info()

    assert info.input_schema["title"] == "BatchInput"
    assert info.output_schema["title"] == "BatchOutput"
    assert info.input_schema["properties"]["repeat"]["type"] == "integer"
    assert info.output_schema["properties"]["tokens"]["items"]["type"] == "string"
    assert tactic.run({"text": "ha", "repeat": 3}) == {
        "tokens": ["ha", "ha", "ha"],
        "count": 3,
    }


def test_tactic_rejects_invalid_dataclass_input():
    with pytest.raises(SchemaError):
        BatchTactic().run({"text": "ha", "repeat": "many"})


def test_protocol_layer_has_no_runtime_or_service_imports():
    leaks: list[str] = []
    for path in sorted(PROTOCOL_ROOT.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for module in _source_imports(tree):
            if module.startswith(FORBIDDEN_PROTOCOL_IMPORT_PREFIXES):
                leaks.append(f"{path.relative_to(ROOT)} imports {module}")

    assert leaks == []


def test_top_level_import_does_not_require_optional_runtime_dependencies(tmp_path):
    _assert_import_while_blocking(
        tmp_path,
        "lllm",
        ("fastapi", "httpx", "pydantic_ai", "uvicorn"),
    )


def _source_imports(tree: ast.AST) -> list[str]:
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = _resolve_import_from(node)
            if module:
                modules.append(module)
    return modules


def _resolve_import_from(node: ast.ImportFrom) -> str:
    if node.level == 0:
        return node.module or ""
    if node.level == 1:
        return f"lllm.protocol.{node.module}" if node.module else "lllm.protocol"
    if node.level == 2:
        return f"lllm.{node.module}" if node.module else "lllm"
    return f"<outside-lllm>.{node.module}" if node.module else "<outside-lllm>"


def _assert_import_while_blocking(
    tmp_path: Path,
    package: str,
    optional_modules: tuple[str, ...],
) -> None:
    code = textwrap.dedent(
        f"""
        import importlib
        import importlib.abc
        import sys

        blocked = {optional_modules!r}

        class BlockOptional(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".", 1)[0] in blocked:
                    raise ModuleNotFoundError(
                        f"blocked optional dependency: {{fullname}}"
                    )
                return None

        sys.meta_path.insert(0, BlockOptional())
        module = importlib.import_module({package!r})
        print(module.__version__)
        """
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(ROOT)
        if not env.get("PYTHONPATH")
        else f"{ROOT}{os.pathsep}{env['PYTHONPATH']}"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
