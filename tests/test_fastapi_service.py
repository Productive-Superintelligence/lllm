import asyncio
import json
from dataclasses import dataclass

import httpx
from pydantic import BaseModel

from lllm import Tactic, endpoint
from lllm.services import create_service_app, create_tactic_app


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

    @endpoint.post("/act")
    async def act(self, input_value, *, context=None):
        return {"acted": input_value.text, "endpoint": context.endpoint}


class StreamTactic(Tactic[str, str]):
    name = "streamer"
    input_type = str
    output_type = str

    def _run(self, input_value, *, context=None):
        return input_value

    def stream(self, input_value, *, context=None):
        yield input_value
        yield input_value.upper()


@dataclass(frozen=True)
class ResponseSnapshot:
    status_code: int
    content: bytes
    headers: dict[str, str]

    def json(self):
        return json.loads(self.content.decode("utf-8"))

    @property
    def text(self):
        return self.content.decode("utf-8")


def request(app, method, path, **kwargs):
    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.request(method, path, **kwargs)
            content = await response.aread()
            return ResponseSnapshot(
                status_code=response.status_code,
                content=content,
                headers=dict(response.headers),
            )

    return asyncio.run(run())


def test_single_tactic_app_accepts_envelope_and_raw_json():
    app = create_tactic_app(EchoTactic())

    envelope = request(
        app,
        "POST",
        "/run",
        json={"input": {"text": "hello"}, "context": {"request_id": "req-1"}},
    )
    raw = request(app, "POST", "/run", json={"text": "world"})

    assert envelope.status_code == 200
    assert envelope.json() == {
        "output": {"text": "HELLO"},
        "request_id": "req-1",
        "tactic": "echo",
    }
    assert raw.status_code == 200
    assert raw.json()["output"] == {"text": "WORLD"}


def test_multi_tactic_service_lists_and_runs_named_tactics():
    app = create_service_app([EchoTactic(), StreamTactic()])

    listed = request(app, "GET", "/tactics")
    response = request(app, "POST", "/tactics/echo/run", json={"input": {"text": "hello"}})

    assert listed.status_code == 200
    assert {item["name"] for item in listed.json()} == {"echo", "streamer"}
    assert response.status_code == 200
    assert response.json()["output"] == {"text": "HELLO"}


def test_stream_endpoint_returns_sse_events():
    app = create_tactic_app(StreamTactic())
    response = request(app, "POST", "/stream", json={"input": "hello"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert '"data": "hello"' in response.text
    assert '"data": "HELLO"' in response.text


def test_custom_endpoint_metadata_mounts_route():
    app = create_tactic_app(EchoTactic())
    response = request(app, "POST", "/act", json={"input": {"text": "go"}})

    assert response.status_code == 200
    assert response.json() == {"acted": "go", "endpoint": "act"}


def test_error_envelope_is_stable():
    app = create_tactic_app(EchoTactic())
    response = request(app, "POST", "/run", json={"input": {"text": 123}})

    assert response.status_code == 500
    detail = response.json()["detail"]["error"]
    assert detail["type"] == "SchemaError"
    assert detail["tactic"] == "echo"
