import asyncio
import json
from dataclasses import dataclass

import httpx
import pytest
from pydantic import BaseModel

from lllm import CallContext, InMemoryProxyLog, ProxyTactic, Tactic, as_tactic, proxy_tactic
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
