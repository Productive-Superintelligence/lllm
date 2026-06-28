import asyncio
import json
from dataclasses import dataclass

import httpx
import pytest
from pydantic import BaseModel

from lllm import (
    CallContext,
    InMemoryProxyLog,
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
