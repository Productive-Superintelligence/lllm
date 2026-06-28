import ast
from pathlib import Path

import pytest
from pydantic import BaseModel

from lllm import CallContext, SchemaError, Tactic, as_tactic


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


def test_protocol_layer_has_no_runtime_or_service_imports():
    leaks: list[str] = []
    for path in sorted(PROTOCOL_ROOT.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for module in _source_imports(tree):
            if module.startswith(FORBIDDEN_PROTOCOL_IMPORT_PREFIXES):
                leaks.append(f"{path.relative_to(ROOT)} imports {module}")

    assert leaks == []


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
