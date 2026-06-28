import pytest
from pydantic import BaseModel

from lllm import CallContext, SchemaError, Tactic, as_tactic


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

    tactic = as_tactic(shout)

    assert tactic.info().input_schema["type"] == "string"
    assert tactic.run("hello", context=CallContext(metadata={"caller": "test"})) == "HELLO"
