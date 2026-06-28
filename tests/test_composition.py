import asyncio

import httpx
import pytest
from pydantic import BaseModel

from lllm import (
    CallContext,
    RemoteTactic,
    Tactic,
    TacticRef,
    TacticRefError,
    TacticResolver,
)
from lllm.services import create_tactic_app


REF = "psi://demo/echo/tactics/echo"


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


def test_tactic_ref_parses_psi_tactic_refs():
    ref = TacticRef(REF)

    assert ref.org == "demo"
    assert ref.package == "echo"
    assert ref.resource_kind == "tactics"
    assert ref.name == "echo"
    assert str(ref) == REF


def test_tactic_ref_rejects_non_tactic_refs():
    with pytest.raises(TacticRefError):
        TacticRef("psi://demo/echo/channels/events")


def test_resolver_calls_in_process_tactic():
    resolver = TacticResolver()
    resolver.register(REF, EchoTactic())

    result = resolver.run(
        REF,
        {"text": "hello"},
        context=CallContext(metadata={"suffix": "!"}),
    )

    assert result == EchoOutput(text="HELLO!")
    assert resolver.refs() == (REF,)


def test_remote_tactic_calls_fastapi_service():
    app = create_tactic_app(EchoTactic())
    remote = RemoteTactic(
        "http://testserver/run",
        name="echo",
        input_type=EchoInput,
        output_type=EchoOutput,
        async_transport=httpx.ASGITransport(app=app),
    )

    result = asyncio.run(
        remote.arun(
            {"text": "hello"},
            context=CallContext(request_id="req-1", metadata={"suffix": "!"}),
        )
    )

    assert result == EchoOutput(text="HELLO!")


def test_resolver_calls_bound_http_tactic():
    app = create_tactic_app(EchoTactic())
    resolver = TacticResolver()
    resolver.bind_url(
        REF,
        "http://testserver/run",
        input_type=EchoInput,
        output_type=EchoOutput,
        async_transport=httpx.ASGITransport(app=app),
    )

    result = asyncio.run(
        resolver.arun(
            REF,
            {"text": "hello"},
            context=CallContext(metadata={"suffix": "?"}),
        )
    )

    assert result == EchoOutput(text="HELLO?")


def test_resolver_loads_url_bindings_from_local_config(tmp_path):
    config_dir = tmp_path / ".psi"
    config_dir.mkdir()
    (config_dir / "config.toml").write_text(
        f"""
[refs."{REF}"]
url = "http://127.0.0.1:8000/tactics/echo"
""".lstrip(),
        encoding="utf-8",
    )

    resolver = TacticResolver.from_config(tmp_path)
    tactic = resolver.resolve(REF)

    assert isinstance(tactic, RemoteTactic)
    assert tactic.url == "http://127.0.0.1:8000/tactics/echo/run"
