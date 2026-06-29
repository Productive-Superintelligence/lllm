import asyncio
import json
from dataclasses import dataclass

import httpx
import pytest
from pydantic import BaseModel, ValidationError

from lllm import (
    CallContext,
    InMemoryProxyLog,
    ProxyRecord,
    ProxyTactic,
    Tactic,
    TacticEvent,
    as_tactic,
    proxy_tactic,
)
from lllm.services import create_tactic_app


class EchoInput(BaseModel):
    text: str


class EchoOutput(BaseModel):
    text: str


class EchoTactic(Tactic[EchoInput, EchoOutput]):
    name = "echo"
    input_type = EchoInput
    output_type = EchoOutput

    def _run(self, input_value, *, context=None):
        return EchoOutput(text=input_value.text.upper())


class StreamTactic(Tactic[str, str]):
    name = "streamer"
    input_type = str
    output_type = str

    def _run(self, input_value, *, context=None):
        return input_value

    def stream(self, input_value, *, context=None):
        yield input_value
        yield input_value.upper()


class EventTactic(Tactic[str, str]):
    name = "events"
    input_type = str
    output_type = str

    def _run(self, input_value, *, context=None):
        return input_value

    async def aevents(self, input_value, *, context=None):
        yield TacticEvent(kind="progress", data=input_value)
        yield TacticEvent.result(input_value.upper())


@dataclass(frozen=True)
class ResponseSnapshot:
    status_code: int
    content: bytes

    def json(self):
        return json.loads(self.content.decode("utf-8"))


def request(app, method, path, **kwargs):
    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.request(method, path, **kwargs)
            return ResponseSnapshot(
                status_code=response.status_code,
                content=await response.aread(),
            )

    return asyncio.run(run())


def test_proxy_tactic_records_success_and_runs_hooks():
    log = InMemoryProxyLog()

    def before(value, context):
        return {"text": f"{value.text}{context.metadata['suffix']}"}

    def after(value, context):
        return {"text": f"{value.text}!"}

    proxy = ProxyTactic(
        EchoTactic(),
        before=before,
        after=after,
        sink=log.append,
        capture_inputs=True,
        capture_outputs=True,
        metadata={"purpose": "test"},
    )

    output = proxy.run(
        {"text": "hi"},
        context=CallContext(request_id="req-1", metadata={"suffix": " there"}),
    )

    assert output == EchoOutput(text="HI THERE!")
    assert proxy.info().runtime_kind == "proxy"
    assert proxy.info().metadata["proxied_tactic"] == "echo"
    assert len(log.records) == 1
    record = log.records[0]
    assert record.request_id == "req-1"
    assert record.proxy == "echo_proxy"
    assert record.tactic == "echo"
    assert record.state == "success"
    assert record.input_value == {"text": "hi there"}
    assert record.output_value == {"text": "HI THERE!"}
    assert record.metadata["context"] == {"suffix": " there"}
    assert record.metadata["proxy"] == {"purpose": "test"}


def test_proxy_tactic_isolates_mutable_metadata():
    log = InMemoryProxyLog()
    proxy_metadata = {"purpose": {"name": "test"}}
    context_metadata = {"suffix": "!", "trace": {"id": "one"}}
    proxy = ProxyTactic(
        EchoTactic(),
        sink=log.append,
        metadata=proxy_metadata,
    )

    info_before = proxy.info()
    proxy_metadata["purpose"]["name"] = "changed"
    proxy.run(
        {"text": "hi"},
        context=CallContext(request_id="req-meta", metadata=context_metadata),
    )
    context_metadata["trace"]["id"] = "two"
    proxy.proxy_metadata["purpose"]["name"] = "internal-change"

    record = log.records[0]
    assert info_before.metadata["purpose"] == {"name": "test"}
    assert proxy.info().metadata["purpose"] == {"name": "test"}
    assert record.metadata["context"]["trace"] == {"id": "one"}
    assert record.metadata["proxy"]["purpose"] == {"name": "test"}


def test_in_memory_proxy_log_isolates_appended_records():
    log = InMemoryProxyLog()
    record = ProxyRecord(
        request_id="req-log",
        proxy="proxy",
        tactic="echo",
        state="success",
        started_at=1.0,
        ended_at=2.0,
        latency_ms=1000.0,
        input_value={"items": [1]},
        output_value={"items": [2]},
        metadata={"labels": ["log"]},
    )

    log.append(record)
    record.input_value["items"].append(10)
    record.output_value["items"].append(20)
    record.metadata["labels"].append("mutated")

    assert log.records[0].input_value == {"items": [1]}
    assert log.records[0].output_value == {"items": [2]}
    assert log.records[0].metadata == {"labels": ["log"]}


def test_proxy_record_isolates_mutable_constructor_inputs():
    input_value = {"items": [1]}
    output_value = {"items": [2]}
    metadata = {"labels": ["record"]}
    record = ProxyRecord(
        request_id="req-record",
        proxy="proxy",
        tactic="echo",
        state="success",
        started_at=1.0,
        ended_at=2.0,
        latency_ms=1000.0,
        input_value=input_value,
        output_value=output_value,
        metadata=metadata,
    )

    input_value["items"].append(10)
    output_value["items"].append(20)
    metadata["labels"].append("mutated")

    assert record.input_value == {"items": [1]}
    assert record.output_value == {"items": [2]}
    assert record.metadata == {"labels": ["record"]}


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ProxyRecord(
            request_id=b"req",
            proxy="proxy",
            tactic="echo",
            state="success",
            started_at=1.0,
            ended_at=2.0,
            latency_ms=1000.0,
        ),
        lambda: ProxyRecord(
            request_id="req",
            proxy=b"proxy",
            tactic="echo",
            state="success",
            started_at=1.0,
            ended_at=2.0,
            latency_ms=1000.0,
        ),
        lambda: ProxyRecord(
            request_id="req",
            proxy="proxy",
            tactic=b"echo",
            state="success",
            started_at=1.0,
            ended_at=2.0,
            latency_ms=1000.0,
        ),
        lambda: ProxyRecord(
            request_id="req",
            proxy="proxy",
            tactic="echo",
            state="failure",
            started_at=1.0,
            ended_at=2.0,
            latency_ms=1000.0,
            error_type=b"ValueError",
        ),
        lambda: ProxyRecord(
            request_id="req",
            proxy="proxy",
            tactic="echo",
            state="failure",
            started_at=1.0,
            ended_at=2.0,
            latency_ms=1000.0,
            error=b"bad",
        ),
    ],
)
def test_proxy_record_rejects_bytes_for_text_fields(factory):
    with pytest.raises(ValidationError):
        factory()


@pytest.mark.parametrize(
    "error_type",
    ("", "   ", ".", "..", "bad type", "bad/type", "bad:type", "bad\\type"),
)
def test_proxy_record_rejects_malformed_error_type_tokens(error_type):
    with pytest.raises(ValidationError):
        ProxyRecord(
            request_id="req",
            proxy="proxy",
            tactic="echo",
            state="failure",
            started_at=1.0,
            ended_at=2.0,
            latency_ms=1000.0,
            error_type=error_type,
        )


@pytest.mark.parametrize(
    "request_id",
    ("", "   ", ".", "..", "bad id", "bad/id", "bad:id", "bad\\id"),
)
def test_proxy_record_rejects_malformed_request_id_tokens(request_id):
    with pytest.raises(ValidationError):
        ProxyRecord(
            request_id=request_id,
            proxy="proxy",
            tactic="echo",
            state="success",
            started_at=1.0,
            ended_at=2.0,
            latency_ms=1000.0,
        )


def test_proxy_tactic_records_failure_and_calls_error_hook():
    errors = []
    log = InMemoryProxyLog()

    def fail(value):
        raise ValueError(f"bad {value}")

    def on_error(exc, context):
        errors.append((type(exc).__name__, context.request_id))

    proxy = proxy_tactic(
        as_tactic(fail),
        on_error=on_error,
        sink=log.append,
        capture_inputs=True,
    )

    with pytest.raises(ValueError, match="bad nope"):
        proxy.run("nope", context=CallContext(request_id="req-2"))

    assert errors == [("ValueError", "req-2")]
    assert len(log.records) == 1
    record = log.records[0]
    assert record.state == "failure"
    assert record.input_value == "nope"
    assert record.error_type == "ValueError"
    assert record.error == "bad nope"


def test_proxy_tactic_streams_and_records_chunks():
    log = InMemoryProxyLog()

    def after(value, context):
        return f"{value}!"

    proxy = ProxyTactic(
        StreamTactic(),
        after=after,
        sink=log.append,
        capture_inputs=True,
        capture_outputs=True,
    )

    output = list(proxy.stream("hi", context=CallContext(request_id="req-stream")))

    assert output == ["hi!", "HI!"]
    assert proxy.supports("stream")
    assert not ProxyTactic(EchoTactic()).supports("stream")
    assert len(log.records) == 1
    record = log.records[0]
    assert record.state == "success"
    assert record.input_value == "hi"
    assert record.output_value == ["hi!", "HI!"]


def test_proxy_tactic_async_streams_and_records_chunks():
    log = InMemoryProxyLog()

    async def collect():
        proxy = ProxyTactic(
            StreamTactic(),
            sink=log.append,
            capture_outputs=True,
        )
        return [
            item
            async for item in proxy.astream(
                "go",
                context=CallContext(request_id="req-astream"),
            )
        ]

    output = asyncio.run(collect())

    assert output == ["go", "GO"]
    assert len(log.records) == 1
    assert log.records[0].request_id == "req-astream"
    assert log.records[0].output_value == ["go", "GO"]


def test_proxy_tactic_preserves_event_only_tactics():
    log = InMemoryProxyLog()

    async def collect():
        proxy = ProxyTactic(
            EventTactic(),
            sink=log.append,
            capture_outputs=True,
        )
        assert proxy.supports("events")
        assert not proxy.supports("stream")
        return [
            event
            async for event in proxy.aevents(
                "go",
                context=CallContext(request_id="req-events"),
            )
        ]

    events = asyncio.run(collect())

    assert [event.kind for event in events] == ["progress", "result"]
    assert [event.data for event in events] == ["go", "GO"]
    assert len(log.records) == 1
    assert log.records[0].request_id == "req-events"
    assert [event["kind"] for event in log.records[0].output_value] == [
        "progress",
        "result",
    ]


def test_proxy_tactic_can_be_served_over_fastapi():
    log = InMemoryProxyLog()
    proxy = ProxyTactic(
        EchoTactic(),
        sink=log.append,
        capture_inputs=True,
        capture_outputs=True,
    )
    app = create_tactic_app(proxy)

    response = request(
        app,
        "POST",
        "/run",
        json={"input": {"text": "hello"}, "context": {"request_id": "req-3"}},
    )

    assert response.status_code == 200
    assert response.json() == {
        "output": {"text": "HELLO"},
        "request_id": "req-3",
        "tactic": "echo_proxy",
    }
    assert len(log.records) == 1
    assert log.records[0].request_id == "req-3"
    assert log.records[0].output_value == {"text": "HELLO"}
