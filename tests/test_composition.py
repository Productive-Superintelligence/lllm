import asyncio

import httpx
import pytest
from pydantic import BaseModel

from lllm import (
    CallContext,
    RemoteTactic,
    RemoteTacticError,
    Tactic,
    TacticEvent,
    TacticRef,
    TacticRefError,
    TacticResolver,
)
from lllm.services import create_service_app, create_tactic_app
from lllm.services.client import _request_envelope


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


class FailingTactic(Tactic[str, str]):
    name = "fail"
    input_type = str
    output_type = str

    def _run(self, input_value, *, context=None):
        raise ValueError(f"cannot handle {input_value}")


class StreamTactic(Tactic[str, str]):
    name = "streamer"
    input_type = str
    output_type = str

    def _run(self, input_value, *, context=None):
        return input_value

    def stream(self, input_value, *, context=None):
        yield input_value
        yield input_value.upper()


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


@pytest.mark.parametrize("value", [None, 123, ""])
def test_tactic_ref_rejects_non_string_or_empty_values(value):
    with pytest.raises(TacticRefError, match="non-empty string"):
        TacticRef(value)
    with pytest.raises(TacticRefError, match="non-empty string"):
        TacticRef.parse(value)


@pytest.mark.parametrize(
    "value",
    [
        "psi://demo/echo/tactics/echo?env=dev",
        "psi://demo/echo/tactics/echo#fragment",
        "psi://demo:bad/echo/tactics/echo",
        "psi://demo/echo:bad/tactics/echo",
        r"psi://demo/echo/tactics/echo\bad",
        "psi://../echo/tactics/echo",
        "psi://demo/./tactics/echo",
        "psi://demo/echo/tactics/..",
        "psi://demo/echo/tactics//echo",
        "psi://demo/echo//tactics/echo",
        "psi://demo/echo/tactics/echo/",
        "psi://demo/   /tactics/echo",
        "psi://demo/echo/tactics/   ",
    ],
)
def test_tactic_ref_rejects_non_resource_url_parts(value):
    with pytest.raises(TacticRefError):
        TacticRef(value)


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


def test_remote_tactic_request_envelope_isolates_mutable_inputs():
    input_value = {"items": ["hello"]}
    context = CallContext(metadata={"labels": ["ctx"]})

    envelope = _request_envelope(input_value, context)
    input_value["items"].append("changed")
    context.metadata["labels"].append("changed")

    assert envelope["input"] == {"items": ["hello"]}
    assert envelope["context"]["metadata"] == {"labels": ["ctx"]}


def test_remote_tactic_fetches_service_info():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/info"
        return httpx.Response(
            200,
            json=EchoTactic().info().model_dump(mode="json"),
        )

    remote = RemoteTactic(
        "http://testserver/run",
        name="echo",
        transport=httpx.MockTransport(handler),
    )

    info = remote.fetch_info()

    assert info.name == "echo"
    assert info.output_schema is not None


def test_remote_tactic_metadata_keeps_normalized_service_url():
    remote = RemoteTactic(
        "http://testserver/tactics/echo",
        name="echo",
        metadata={"url": "http://spoofed", "labels": ["remote"]},
    )

    info = remote.info()

    assert remote.url == "http://testserver/tactics/echo/run"
    assert info.metadata["url"] == "http://testserver/tactics/echo/run"
    assert info.metadata["labels"] == ["remote"]


@pytest.mark.parametrize(
    "url",
    [
        None,
        123,
        "",
        "   ",
        "testserver/run",
        "/run",
        "ftp://testserver/run",
        "http://test server/run",
    ],
)
def test_remote_tactic_rejects_malformed_service_urls(url):
    with pytest.raises(ValueError, match="url"):
        RemoteTactic(url)  # type: ignore[arg-type]


def test_remote_tactic_async_fetches_service_info():
    app = create_tactic_app(EchoTactic())
    remote = RemoteTactic(
        "http://testserver/run",
        name="echo",
        async_transport=httpx.ASGITransport(app=app),
    )

    info = asyncio.run(remote.afetch_info())

    assert info.name == "echo"
    assert "stream" not in info.capabilities


def test_remote_tactic_normalizes_direct_info_urls():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/tactics/echo/info"
        return httpx.Response(200, json=EchoTactic().info().model_dump(mode="json"))

    remote = RemoteTactic(
        "http://testserver/tactics/echo/info",
        transport=httpx.MockTransport(handler),
    )

    assert remote.tactic_name == "echo"
    assert remote.url == "http://testserver/tactics/echo/run"
    assert remote.stream_url == "http://testserver/tactics/echo/stream"
    assert remote.info_url == "http://testserver/tactics/echo/info"
    assert remote.fetch_info().name == "echo"


def test_remote_tactic_fetches_multi_tactic_service_info():
    app = create_service_app({"echo": EchoTactic(), "streamer": StreamTactic()})
    remote = RemoteTactic(
        "http://testserver/tactics/echo/run",
        async_transport=httpx.ASGITransport(app=app),
    )

    info = asyncio.run(remote.afetch_info())

    assert info.name == "echo"
    assert remote.info_url == "http://testserver/tactics/echo/info"


def test_remote_tactic_preserves_structured_service_errors():
    app = create_tactic_app(FailingTactic())
    remote = RemoteTactic(
        "http://testserver/run",
        name="fail",
        input_type=str,
        output_type=str,
        async_transport=httpx.ASGITransport(app=app),
    )

    with pytest.raises(RemoteTacticError) as exc_info:
        asyncio.run(
            remote.arun(
                "boom",
                context=CallContext(request_id="req-error"),
            )
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.error_type == "ValueError"
    assert exc_info.value.message == "cannot handle boom"
    assert exc_info.value.tactic == "fail"
    assert exc_info.value.endpoint == "run"
    assert exc_info.value.request_id == "req-error"


def test_remote_tactic_preserves_protocol_error_status():
    app = create_tactic_app(EchoTactic())
    remote = RemoteTactic(
        "http://testserver/run",
        name="echo",
        async_transport=httpx.ASGITransport(app=app),
    )

    with pytest.raises(RemoteTacticError) as exc_info:
        asyncio.run(
            remote.arun(
                {"text": 123},
                context=CallContext(request_id="req-schema"),
            )
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.error_type == "SchemaError"
    assert exc_info.value.tactic == "echo"
    assert exc_info.value.request_id == "req-schema"


def test_remote_tactic_preserves_plain_text_service_errors():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/run"
        return httpx.Response(503, text="upstream unavailable")

    remote = RemoteTactic(
        "http://testserver/run",
        name="echo",
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(RemoteTacticError) as exc_info:
        remote.run("hello")

    assert exc_info.value.status_code == 503
    assert exc_info.value.error_type is None
    assert exc_info.value.message == "upstream unavailable"
    assert exc_info.value.detail == "upstream unavailable"


def test_remote_tactic_reports_invalid_success_json():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/run"
        return httpx.Response(200, text="{bad")

    remote = RemoteTactic(
        "http://testserver/run",
        name="echo",
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(RemoteTacticError) as exc_info:
        remote.run("hello")

    assert exc_info.value.status_code == 200
    assert exc_info.value.error_type == "InvalidResponse"
    assert exc_info.value.endpoint == "run"
    assert "not valid JSON" in exc_info.value.message
    assert exc_info.value.detail == "{bad"


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (httpx.Response(200, text="{bad"), "not valid JSON"),
        (httpx.Response(200, json={"name": 123}), "TacticInfo schema"),
    ],
)
def test_remote_tactic_reports_invalid_info_responses(response, expected):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/info"
        return response

    remote = RemoteTactic(
        "http://testserver/run",
        name="echo",
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(RemoteTacticError) as exc_info:
        remote.fetch_info()

    assert exc_info.value.status_code == 200
    assert exc_info.value.error_type == "InvalidResponse"
    assert exc_info.value.endpoint == "info"
    assert expected in exc_info.value.message


def test_remote_tactic_streams_fastapi_sse_events():
    app = create_tactic_app(StreamTactic())
    remote = RemoteTactic(
        "http://testserver/run",
        name="streamer",
        input_type=str,
        output_type=str,
        async_transport=httpx.ASGITransport(app=app),
    )

    async def collect():
        return [
            item
            async for item in remote.astream(
                "hello",
                context=CallContext(request_id="req-stream"),
            )
        ]

    assert asyncio.run(collect()) == ["hello", "HELLO"]


def test_remote_tactic_preserves_stream_events():
    app = create_tactic_app(StreamTactic())
    remote = RemoteTactic(
        "http://testserver/run",
        name="streamer",
        input_type=str,
        output_type=str,
        async_transport=httpx.ASGITransport(app=app),
    )

    async def collect():
        return [
            event
            async for event in remote.aevents(
                "hello",
                context=CallContext(request_id="req-event-stream"),
            )
        ]

    events = asyncio.run(collect())

    assert [event.data for event in events] == ["hello", "HELLO"]
    assert all(isinstance(event, TacticEvent) for event in events)
    assert all(event.kind == "message" for event in events)


def test_remote_tactic_sync_streams_sse_events():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/stream"
        assert request.read()
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                b'data: {"kind": "message", "data": "hello"}\n\n'
                b'data: {"kind": "message", "data": "HELLO"}\n\n'
            ),
        )

    remote = RemoteTactic(
        "http://testserver/run",
        name="streamer",
        input_type=str,
        output_type=str,
        transport=httpx.MockTransport(handler),
    )

    assert list(remote.stream("hello")) == ["hello", "HELLO"]


def test_remote_tactic_preserves_raw_json_object_stream_chunks():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/stream"
        assert request.read()
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=b'data: {"kind": "raw", "text": "hello"}\n\n',
        )

    remote = RemoteTactic(
        "http://testserver/run",
        name="streamer",
        input_type=str,
        output_type=dict,
        transport=httpx.MockTransport(handler),
    )

    assert list(remote.stream("hello")) == [{"kind": "raw", "text": "hello"}]


def test_remote_tactic_reports_invalid_sse_event_envelopes():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/stream"
        assert await request.aread()
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                b'data: {"kind": "message", "data": "hello", '
                b'"timestamp": "not-a-number"}\n\n'
            ),
        )

    remote = RemoteTactic(
        "http://testserver/run",
        name="streamer",
        input_type=str,
        output_type=str,
        async_transport=httpx.MockTransport(handler),
    )

    async def collect():
        return [event async for event in remote.aevents("hello")]

    events = asyncio.run(collect())

    assert len(events) == 1
    assert events[0].kind == "error"
    assert events[0].data == {"message": "Invalid SSE event envelope."}
    assert events[0].metadata["payload"]["data"] == "hello"
    assert events[0].metadata["errors"]


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

[refs."psi://demo/echo/services/api"]
url = "http://127.0.0.1:8000"

[refs."psi://demo/echo/channels/events"]
store = ".sssn"
""".lstrip(),
        encoding="utf-8",
    )

    resolver = TacticResolver.from_config(tmp_path)
    tactic = resolver.resolve(REF)

    assert isinstance(tactic, RemoteTactic)
    assert tactic.url == "http://127.0.0.1:8000/tactics/echo/run"
    assert resolver.refs() == (REF,)


def test_resolver_trims_url_bindings_from_local_config(tmp_path):
    config_dir = tmp_path / ".psi"
    config_dir.mkdir()
    (config_dir / "config.toml").write_text(
        f"""
[refs."{REF}"]
url = "  http://127.0.0.1:8000/tactics/echo  "
""".lstrip(),
        encoding="utf-8",
    )

    resolver = TacticResolver.from_config(tmp_path)
    tactic = resolver.resolve(REF)

    assert isinstance(tactic, RemoteTactic)
    assert tactic.url == "http://127.0.0.1:8000/tactics/echo/run"


def test_resolver_rejects_malformed_config_paths():
    for path in ("   ", 123):
        with pytest.raises(ValueError, match="config path"):
            TacticResolver.from_config(path)  # type: ignore[arg-type]


def test_resolver_rejects_malformed_url_bindings_from_local_config(tmp_path):
    config_dir = tmp_path / ".psi"
    config_dir.mkdir()
    (config_dir / "config.toml").write_text(
        """
[refs."not-a-ref"]
url = "http://127.0.0.1:8000/tactics/echo"
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(TacticRefError, match="psi://"):
        TacticResolver.from_config(tmp_path)


def test_resolver_rejects_invalid_tactic_url_values_from_local_config(tmp_path):
    for index, url_value in enumerate(("123", "false", '""', '"   "'), start=1):
        config_dir = tmp_path / f"workspace-{index}" / ".psi"
        config_dir.mkdir(parents=True)
        (config_dir / "config.toml").write_text(
            f"""
[refs."{REF}"]
url = {url_value}
""".lstrip(),
            encoding="utf-8",
        )

        with pytest.raises(TacticRefError, match="non-empty string"):
            TacticResolver.from_config(config_dir.parent)
