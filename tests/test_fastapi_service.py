import asyncio
import json
from dataclasses import dataclass

import httpx
import pytest
from pydantic import BaseModel

from lllm import Tactic, endpoint
from lllm.services import create_service_app, create_tactic_app
from lllm.services.fastapi import ErrorDetail, RunRequest, RunResponse


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

    @endpoint.put("/state")
    async def update_state(self, input_value, *, context=None):
        return {"updated": input_value.text, "endpoint": context.endpoint}


class StreamTactic(Tactic[str, str]):
    name = "streamer"
    input_type = str
    output_type = str

    def _run(self, input_value, *, context=None):
        return input_value

    def stream(self, input_value, *, context=None):
        yield input_value
        yield input_value.upper()


class FailingStreamTactic(Tactic[str, str]):
    name = "failing-streamer"
    input_type = str
    output_type = str

    def _run(self, input_value, *, context=None):
        return input_value

    def stream(self, input_value, *, context=None):
        yield input_value
        raise ValueError("stream failed")


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


def test_service_dto_models_isolate_mutable_constructor_inputs():
    input_value = {"items": [1]}
    task_value = {"task": [1]}
    context = {"metadata": {"labels": ["request"]}}
    request = RunRequest(input=input_value, task=task_value, context=context)
    output = {"items": [2]}
    response = RunResponse(output=output, request_id="req", tactic="echo")
    metadata = {"labels": ["error"]}
    error = ErrorDetail(type="ValueError", message="bad", metadata=metadata)

    input_value["items"].append(10)
    task_value["task"].append(10)
    context["metadata"]["labels"].append("mutated")
    output["items"].append(20)
    metadata["labels"].append("mutated")

    assert request.input == {"items": [1]}
    assert request.task == {"task": [1]}
    assert request.context == {"metadata": {"labels": ["request"]}}
    assert request.value == {"items": [1]}
    assert response.output == {"items": [2]}
    assert error.metadata == {"labels": ["error"]}


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


def test_service_rejects_path_control_tactic_route_names():
    bad_names = ("", ".", "..", "bad/name", r"bad\name", "bad:name", None, 123)

    for name in bad_names:
        with pytest.raises(ValueError, match="tactic.name"):
            create_service_app({name: EchoTactic()})


def test_service_rejects_duplicate_sequence_tactic_route_names():
    with pytest.raises(ValueError, match="Duplicate tactic.name"):
        create_service_app([EchoTactic(), EchoTactic()])


def test_stream_endpoint_returns_sse_events():
    app = create_tactic_app(StreamTactic())
    response = request(app, "POST", "/stream", json={"input": "hello"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert '"data": "hello"' in response.text
    assert '"data": "HELLO"' in response.text


def test_stream_endpoint_reports_iterator_errors_as_sse_events():
    app = create_tactic_app(FailingStreamTactic())
    response = request(
        app,
        "POST",
        "/stream",
        json={"input": "hello", "context": {"request_id": "req-stream-error"}},
    )

    chunks = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]

    assert response.status_code == 200
    assert chunks[0]["kind"] == "message"
    assert chunks[0]["data"] == "hello"
    assert chunks[1]["kind"] == "error"
    assert chunks[1]["data"] == {"message": "stream failed"}
    assert chunks[1]["metadata"] == {
        "error_type": "ValueError",
        "tactic": "failing-streamer",
        "endpoint": "stream",
        "request_id": "req-stream-error",
    }


def test_stream_endpoint_reports_unsupported_tactic_stably():
    app = create_tactic_app(EchoTactic())
    response = request(app, "POST", "/stream", json={"input": {"text": "hello"}})

    assert response.status_code == 400
    detail = response.json()["detail"]["error"]
    assert detail["type"] == "TacticUnsupportedError"
    assert detail["endpoint"] == "stream"


def test_custom_endpoint_metadata_mounts_route():
    app = create_tactic_app(EchoTactic())
    response = request(app, "POST", "/act", json={"input": {"text": "go"}})
    put_response = request(app, "PUT", "/state", json={"input": {"text": "ready"}})

    assert response.status_code == 200
    assert response.json() == {"acted": "go", "endpoint": "act"}
    assert put_response.status_code == 200
    assert put_response.json() == {"updated": "ready", "endpoint": "update_state"}


def test_error_envelope_is_stable():
    app = create_tactic_app(EchoTactic())
    response = request(app, "POST", "/run", json={"input": {"text": 123}})

    assert response.status_code == 400
    detail = response.json()["detail"]["error"]
    assert detail["type"] == "SchemaError"
    assert detail["tactic"] == "echo"


def test_invalid_request_context_returns_stable_error_envelope():
    app = create_tactic_app(EchoTactic())

    bad_context = request(
        app,
        "POST",
        "/run",
        json={"input": {"text": "hello"}, "context": "bad"},
    )
    bad_metadata = request(
        app,
        "POST",
        "/act",
        json={"input": {"text": "go"}, "context": {"metadata": []}},
    )
    bad_tags = request(
        app,
        "POST",
        "/run",
        json={"input": {"text": "hello"}, "context": {"tags": []}},
    )
    bad_request_id = request(
        app,
        "POST",
        "/run",
        json={"input": {"text": "hello"}, "context": {"request_id": 0}},
    )
    bad_tactic_ref = request(
        app,
        "POST",
        "/run",
        json={"input": {"text": "hello"}, "context": {"tactic_ref": 0}},
    )

    assert bad_context.status_code == 400
    context_detail = bad_context.json()["detail"]["error"]
    assert context_detail["type"] == "SchemaError"
    assert context_detail["tactic"] == "echo"
    assert context_detail["endpoint"] == "run"
    assert "context" in context_detail["message"]
    assert bad_metadata.status_code == 400
    metadata_detail = bad_metadata.json()["detail"]["error"]
    assert metadata_detail["type"] == "SchemaError"
    assert metadata_detail["endpoint"] == "act"
    assert "metadata" in metadata_detail["message"]
    assert bad_tags.status_code == 400
    tags_detail = bad_tags.json()["detail"]["error"]
    assert tags_detail["type"] == "SchemaError"
    assert tags_detail["endpoint"] == "run"
    assert "tags" in tags_detail["message"]
    assert bad_request_id.status_code == 400
    request_id_detail = bad_request_id.json()["detail"]["error"]
    assert request_id_detail["type"] == "SchemaError"
    assert request_id_detail["endpoint"] == "run"
    assert "context" in request_id_detail["message"]
    assert bad_tactic_ref.status_code == 400
    tactic_ref_detail = bad_tactic_ref.json()["detail"]["error"]
    assert tactic_ref_detail["type"] == "SchemaError"
    assert tactic_ref_detail["endpoint"] == "run"
    assert "context" in tactic_ref_detail["message"]


def test_single_tactic_openapi_includes_default_and_custom_routes():
    app = create_tactic_app(EchoTactic())
    schema = app.openapi()

    assert "/run" in schema["paths"]
    assert "/stream" in schema["paths"]
    assert "/info" in schema["paths"]
    assert "/act" in schema["paths"]
    assert "/state" in schema["paths"]
    assert schema["paths"]["/act"]["post"]["summary"] == "Act"
    assert schema["paths"]["/state"]["put"]["summary"] == "Update State"


def test_multi_tactic_openapi_includes_portable_routes():
    app = create_service_app([EchoTactic(), StreamTactic()])
    schema = app.openapi()

    assert "/tactics" in schema["paths"]
    assert "/tactics/{name}/info" in schema["paths"]
    assert "/tactics/{name}/run" in schema["paths"]
    assert "/tactics/{name}/stream" in schema["paths"]
