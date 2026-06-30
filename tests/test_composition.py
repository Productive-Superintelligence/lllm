import asyncio
import json

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
from lllm.services.client import _aiter_sse_events, _iter_sse_events, _request_envelope


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
        "psi://demo org/echo/tactics/echo",
        "psi://demo/echo pkg/tactics/echo",
        "psi://demo/echo/tact ics/echo",
        "psi://demo/echo/tactics/echo name",
        "psi://demo%2Forg/echo/tactics/echo",
        "psi://demo/echo%2Fpkg/tactics/echo",
        "psi://demo/echo/tactics/echo%2Fname",
        "psi://demo/echo/tactics/echo%5Cname",
        "psi://demo/echo/tactics/%2E%2E",
        "psi://demo/echo/tactics/echo%20name",
        "psi://demo/echo/tactics/echo%3Aname",
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


def test_resolver_register_rejects_non_tactic_objects():
    resolver = TacticResolver()

    for tactic in (object(), lambda value: value, None):
        with pytest.raises(TypeError, match="Tactic"):
            resolver.register(REF, tactic)  # type: ignore[arg-type]

    assert resolver.refs() == ()


def test_resolver_bind_url_rejects_malformed_service_urls():
    resolver = TacticResolver()

    for url in ("service", "/service", "ftp://service", "http://"):
        with pytest.raises(TacticRefError, match="absolute http"):
            resolver.bind_url(REF, url)

    assert resolver.refs() == ()


def test_resolver_bind_url_preserves_metadata_with_canonical_ref():
    resolver = TacticResolver()
    resolver.bind_url(
        REF,
        "http://testserver/tactics/echo",
        metadata={"ref": "spoofed", "label": "direct"},
    )

    tactic = resolver.resolve(REF)
    metadata = tactic.info().metadata

    assert metadata["ref"] == REF
    assert metadata["url"] == "http://testserver/tactics/echo/run"
    assert metadata["label"] == "direct"


def test_resolver_bind_url_rejects_non_mapping_metadata():
    resolver = TacticResolver()

    with pytest.raises(TacticRefError, match="metadata"):
        resolver.bind_url(  # type: ignore[arg-type]
            REF,
            "http://testserver/tactics/echo",
            metadata=[],
        )

    assert resolver.refs() == ()


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


def test_remote_tactic_sync_run_sends_one_request():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/run"
        requests.append(json.loads(request.read().decode("utf-8")))
        return httpx.Response(200, json={"output": {"text": "HELLO!"}})

    remote = RemoteTactic(
        "http://testserver/run",
        name="echo",
        input_type=EchoInput,
        output_type=EchoOutput,
        transport=httpx.MockTransport(handler),
    )

    result = remote.run(
        {"text": "hello"},
        context=CallContext(request_id="req-sync", metadata={"suffix": "!"}),
    )

    assert result == EchoOutput(text="HELLO!")
    assert len(requests) == 1
    assert requests[0]["input"] == {"text": "hello"}
    assert requests[0]["context"]["request_id"] == "req-sync"
    assert requests[0]["context"]["metadata"] == {"suffix": "!"}


def test_remote_tactic_request_envelope_isolates_mutable_inputs():
    input_value = {"items": ["hello"]}
    context = CallContext(metadata={"labels": ["ctx"]})

    envelope = _request_envelope(input_value, context)
    input_value["items"].append("changed")
    context.metadata["labels"].append("changed")

    assert envelope["input"] == {"items": ["hello"]}
    assert envelope["context"]["metadata"] == {"labels": ["ctx"]}


def test_remote_tactic_request_envelope_rejects_malformed_context():
    with pytest.raises(TypeError, match="CallContext"):
        _request_envelope(  # type: ignore[arg-type]
            {"text": "hello"},
            {"metadata": {}},
        )


def test_remote_tactic_rejects_malformed_context_without_request():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected request: {request.url}")

    remote = RemoteTactic(
        "http://testserver/run",
        name="echo",
        transport=httpx.MockTransport(handler),
        async_transport=httpx.MockTransport(handler),
    )

    with pytest.raises(TypeError, match="CallContext"):
        remote.run("hello", context={"metadata": {}})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="CallContext"):
        list(  # type: ignore[arg-type]
            remote.stream("hello", context={"metadata": {}})
        )

    async def collect_errors():
        with pytest.raises(TypeError, match="CallContext"):
            await remote.arun(  # type: ignore[arg-type]
                "hello",
                context={"metadata": {}},
            )
        with pytest.raises(TypeError, match="CallContext"):
            [
                event
                async for event in remote.aevents(
                    "hello",
                    context={"metadata": {}},  # type: ignore[arg-type]
                )
            ]

    asyncio.run(collect_errors())


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


@pytest.mark.parametrize("metadata", [[], [("labels", ["remote"])], "bad", 123])
def test_remote_tactic_rejects_non_mapping_metadata(metadata):
    with pytest.raises(TypeError, match="metadata"):
        RemoteTactic(  # type: ignore[arg-type]
            "http://testserver/run",
            metadata=metadata,
        )


@pytest.mark.parametrize(
    "timeout",
    [False, True, "1", 0, -1, float("inf"), float("nan")],
)
def test_remote_tactic_rejects_invalid_timeouts(timeout):
    with pytest.raises(ValueError, match="timeout"):
        RemoteTactic(  # type: ignore[arg-type]
            "http://testserver/run",
            timeout=timeout,
        )


def test_remote_tactic_accepts_positive_timeout_or_none():
    remote = RemoteTactic("http://testserver/run", timeout=1)
    no_timeout = RemoteTactic("http://testserver/run", timeout=None)

    assert remote.timeout == 1.0
    assert no_timeout.timeout is None


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
        " http://testserver/run ",
        "http://test server/run",
    ],
)
def test_remote_tactic_rejects_malformed_service_urls(url):
    with pytest.raises(ValueError, match="url"):
        RemoteTactic(url)  # type: ignore[arg-type]


@pytest.mark.parametrize("name", ["", "   "])
def test_remote_tactic_rejects_explicit_blank_names(name):
    with pytest.raises(ValueError, match="name"):
        RemoteTactic("http://testserver/run", name=name)


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


@pytest.mark.parametrize(
    ("field_name", "kwargs"),
    [
        ("error_type", {"error_type": b"ValueError"}),
        ("message", {"message": b"bad"}),
        ("tactic", {"tactic": b"echo"}),
        ("endpoint", {"endpoint": b"run"}),
        ("request_id", {"request_id": b"req"}),
    ],
)
def test_remote_tactic_error_rejects_bytes_for_text_fields(field_name, kwargs):
    with pytest.raises(TypeError, match=field_name):
        RemoteTacticError(500, **kwargs)


@pytest.mark.parametrize(
    "error_type",
    (
        "",
        "   ",
        ".",
        "..",
        "bad type",
        "bad/type",
        "bad:type",
        "bad\\type",
        "bad%2Ftype",
    ),
)
def test_remote_tactic_error_rejects_malformed_error_type_tokens(error_type):
    with pytest.raises(ValueError, match="error_type"):
        RemoteTacticError(500, error_type=error_type)


@pytest.mark.parametrize(
    "request_id",
    (
        "",
        "   ",
        ".",
        "..",
        "bad id",
        "bad/id",
        "bad:id",
        "bad\\id",
        "bad%2Fid",
    ),
)
def test_remote_tactic_error_rejects_malformed_request_id_tokens(request_id):
    with pytest.raises(ValueError, match="request_id"):
        RemoteTacticError(500, request_id=request_id)


def test_remote_tactic_ignores_non_string_error_payload_fields():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/run"
        return httpx.Response(
            500,
            json={
                "detail": {
                    "error": {
                        "type": 123,
                        "message": ["bad"],
                        "tactic": False,
                        "endpoint": {"path": "run"},
                        "request_id": 7,
                    }
                }
            },
        )

    remote = RemoteTactic(
        "http://testserver/run",
        name="echo",
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(RemoteTacticError) as exc_info:
        remote.run("hello")

    assert exc_info.value.status_code == 500
    assert exc_info.value.error_type is None
    assert exc_info.value.message is None
    assert exc_info.value.tactic is None
    assert exc_info.value.endpoint is None
    assert exc_info.value.request_id is None


def test_remote_tactic_ignores_malformed_error_payload_tokens():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/run"
        return httpx.Response(
            500,
            json={
                "detail": {
                    "error": {
                        "type": "bad type",
                        "message": "still readable",
                        "request_id": "bad id",
                    }
                }
            },
        )

    remote = RemoteTactic(
        "http://testserver/run",
        name="echo",
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(RemoteTacticError) as exc_info:
        remote.run("hello")

    assert exc_info.value.status_code == 500
    assert exc_info.value.error_type is None
    assert exc_info.value.message == "still readable"
    assert exc_info.value.request_id is None


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


def test_sse_parser_reports_non_text_stream_lines():
    events = list(
        _iter_sse_events(
            [
                object(),
                'data: {"kind": "message", "data": "ok"}',
                "",
            ]
        )
    )

    assert [event.kind for event in events] == ["error", "message"]
    assert events[0].data == {"message": "Invalid SSE stream line."}
    assert events[0].metadata["line_type"] == "object"
    assert events[0].metadata["errors"]
    assert events[1].data == "ok"


def test_async_sse_parser_reports_non_text_stream_lines():
    async def lines():
        yield object()
        yield 'data: {"kind": "message", "data": "ok"}'
        yield ""

    async def collect():
        return [event async for event in _aiter_sse_events(lines())]

    events = asyncio.run(collect())

    assert [event.kind for event in events] == ["error", "message"]
    assert events[0].data == {"message": "Invalid SSE stream line."}
    assert events[0].metadata["line_type"] == "object"
    assert events[0].metadata["errors"]
    assert events[1].data == "ok"


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


def test_resolver_loads_url_binding_metadata_from_local_config(tmp_path):
    config_dir = tmp_path / ".psi"
    config_dir.mkdir()
    (config_dir / "config.toml").write_text(
        f"""
[refs."{REF}"]
url = "http://127.0.0.1:8000/tactics/echo"
label = "legacy-extra"
tag = "legacy"
ref = "spoofed-extra"

[refs."{REF}".metadata]
tag = "explicit"
ref = "spoofed-table"
url = "http://spoofed"
headers = {{ x_policy = "demo" }}
""".lstrip(),
        encoding="utf-8",
    )

    resolver = TacticResolver.from_config(tmp_path)
    tactic = resolver.resolve(REF)
    metadata = tactic.info().metadata

    assert metadata == {
        "label": "legacy-extra",
        "tag": "explicit",
        "headers": {"x_policy": "demo"},
        "ref": REF,
        "url": "http://127.0.0.1:8000/tactics/echo/run",
    }


def test_resolver_loaded_config_metadata_is_isolated(tmp_path):
    config_dir = tmp_path / ".psi"
    config_dir.mkdir()
    (config_dir / "config.toml").write_text(
        f"""
[refs."{REF}"]
url = "http://127.0.0.1:8000/tactics/echo"

[refs."{REF}".metadata]
headers = {{ x_policy = "demo" }}
labels = ["alpha", "beta"]
""".lstrip(),
        encoding="utf-8",
    )

    resolver = TacticResolver.from_config(tmp_path)
    metadata = resolver.resolve(REF).info().metadata
    metadata["headers"]["x_policy"] = "changed"
    metadata["labels"].append("changed")

    assert resolver.resolve(REF).info().metadata == {
        "headers": {"x_policy": "demo"},
        "labels": ["alpha", "beta"],
        "ref": REF,
        "url": "http://127.0.0.1:8000/tactics/echo/run",
    }


def test_resolver_rejects_non_table_ref_metadata_from_local_config(tmp_path):
    config_dir = tmp_path / ".psi"
    config_dir.mkdir()
    (config_dir / "config.toml").write_text(
        f"""
[refs."{REF}"]
url = "http://127.0.0.1:8000/tactics/echo"
metadata = "bad"
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(TacticRefError, match="metadata"):
        TacticResolver.from_config(tmp_path)


def test_resolver_rejects_whitespace_bearing_url_bindings_from_local_config(tmp_path):
    config_dir = tmp_path / ".psi"
    config_dir.mkdir()
    (config_dir / "config.toml").write_text(
        f"""
[refs."{REF}"]
url = "  http://127.0.0.1:8000/tactics/echo  "
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(TacticRefError, match="whitespace"):
        TacticResolver.from_config(tmp_path)


def test_resolver_rejects_malformed_config_paths():
    for path in ("   ", " ./config.toml ", 123):
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


@pytest.mark.parametrize(
    "ref",
    [
        "http://demo/echo/channels/events",
        "psi://demo/echo/channels/events?env=dev",
        "psi://demo/echo/channels/events#latest",
        "psi://demo/echo/channels/bad name",
        "psi://demo/echo/channels/events%2Fhidden",
        "psi://demo/echo/widgets/item",
    ],
)
def test_resolver_validates_ignored_non_tactic_config_refs(tmp_path, ref):
    config_dir = tmp_path / "workspace" / ".psi"
    config_dir.mkdir(parents=True)
    (config_dir / "config.toml").write_text(
        f"""
[refs."{ref}"]
store = ".sssn"
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(TacticRefError):
        TacticResolver.from_config(config_dir.parent)


@pytest.mark.parametrize(
    ("resource_kind", "target_line"),
    [
        ("schemas", 'path = "schemas/input.json"'),
        ("services", 'url = "http://127.0.0.1:8000"'),
        ("channels", 'store = ".sssn"'),
        ("snapshots", 'store = ".sssn"'),
        ("runs", 'path = "runs/latest.json"'),
        ("configs", 'path = ".psi/config.toml"'),
        ("docs", 'path = "docs/guide.md"'),
        ("examples", 'path = "examples/demo.py"'),
        ("assets", 'path = "assets/logo.svg"'),
    ],
)
def test_resolver_ignores_known_non_tactic_config_sections(
    tmp_path,
    resource_kind,
    target_line,
):
    config_dir = tmp_path / resource_kind / ".psi"
    config_dir.mkdir(parents=True)
    (config_dir / "config.toml").write_text(
        f"""
[refs."psi://demo/echo/{resource_kind}/local"]
{target_line}
""".lstrip(),
        encoding="utf-8",
    )

    resolver = TacticResolver.from_config(config_dir.parent)

    assert resolver.refs() == ()


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


@pytest.mark.parametrize(
    "extra_target",
    ['store = ".sssn"', 'path = "tactic.py"', 'object = "local"'],
)
def test_resolver_rejects_ambiguous_tactic_url_targets_from_local_config(
    tmp_path,
    extra_target,
):
    config_dir = tmp_path / "workspace" / ".psi"
    config_dir.mkdir(parents=True)
    (config_dir / "config.toml").write_text(
        f"""
[refs."{REF}"]
url = "http://127.0.0.1:8000/tactics/echo"
{extra_target}
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(TacticRefError, match="only one concrete target"):
        TacticResolver.from_config(config_dir.parent)


def test_resolver_rejects_malformed_tactic_url_targets_from_local_config(tmp_path):
    for index, url in enumerate(("service", "/service", "ftp://service", "http://")):
        config_dir = tmp_path / f"workspace-url-{index}" / ".psi"
        config_dir.mkdir(parents=True)
        (config_dir / "config.toml").write_text(
            f"""
[refs."{REF}"]
url = "{url}"
""".lstrip(),
            encoding="utf-8",
        )

        with pytest.raises(TacticRefError, match="absolute http"):
            TacticResolver.from_config(config_dir.parent)
