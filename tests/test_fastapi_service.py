import asyncio
import json
from dataclasses import dataclass

import httpx
import pytest
from pydantic import BaseModel, ValidationError

from lllm import Tactic, endpoint
from lllm.runtimes import PydanticAITactic
from lllm.services import create_service_app, create_tactic_app
from lllm.services.endpoints import EndpointSpec, custom_endpoints
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


class ServicePydanticResult:
    def __init__(self, output):
        self.output = output


class ServicePydanticAgent:
    name = "pydantic-service"
    output_type = EchoOutput

    def __init__(self):
        self.seen_task = None
        self.seen_metadata = None

    def run_sync(self, task, *, metadata=None, **kwargs):
        self.seen_task = dict(task)
        self.seen_metadata = dict(metadata or {})
        return ServicePydanticResult(
            EchoOutput(
                text=f"{task['text'].upper()}:{self.seen_metadata['lllm_trace_id']}",
            )
        )


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


@pytest.mark.parametrize(
    "factory",
    [
        lambda: RunResponse(output={}, request_id=b"req", tactic="echo"),
        lambda: RunResponse(output={}, request_id="req", tactic=b"echo"),
        lambda: ErrorDetail(type=b"ValueError", message="bad"),
        lambda: ErrorDetail(type="ValueError", message=b"bad"),
        lambda: ErrorDetail(type="ValueError", message="bad", tactic=b"echo"),
        lambda: ErrorDetail(type="ValueError", message="bad", endpoint=b"/run"),
        lambda: ErrorDetail(type="ValueError", message="bad", request_id=b"req"),
    ],
)
def test_service_dto_models_reject_bytes_for_string_fields(factory):
    with pytest.raises(ValidationError):
        factory()


@pytest.mark.parametrize(
    "value",
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
def test_service_error_detail_rejects_malformed_type_tokens(value):
    with pytest.raises(ValidationError):
        ErrorDetail(type=value, message="bad")


@pytest.mark.parametrize(
    "value",
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
def test_service_dto_models_reject_malformed_request_id_tokens(value):
    with pytest.raises(ValidationError):
        RunResponse(output={}, request_id=value, tactic="echo")
    with pytest.raises(ValidationError):
        ErrorDetail(type="ValueError", message="bad", request_id=value)


def test_service_error_detail_allows_common_type_tokens():
    response = RunResponse(output={}, request_id="req-1", tactic="echo")
    error = ErrorDetail(type="InvalidResponse", message="bad", request_id="req.1")

    assert response.request_id == "req-1"
    assert error.type == "InvalidResponse"
    assert error.request_id == "req.1"


def test_endpoint_decorator_normalizes_relative_paths():
    class RelativeEndpointTactic(EchoTactic):
        @endpoint.post(
            "custom-act",
            name="custom_act",
            mode="run",
            tags=("policy",),
        )
        async def custom_act(self, input_value, *, context=None):
            return {"acted": input_value.text}

    spec, _method = next(
        item
        for item in custom_endpoints(RelativeEndpointTactic())
        if item[0].name == "custom_act"
    )

    assert spec.path == "/custom-act"
    assert spec.name == "custom_act"
    assert spec.mode == "run"
    assert spec.tags == ("policy",)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: endpoint.post(None),
        lambda: endpoint.post(123),
        lambda: endpoint.post(""),
        lambda: endpoint.post("bad path"),
        lambda: endpoint.post("/act%2Fextra"),
        lambda: endpoint.post("/act?mode=fast"),
        lambda: endpoint.post("/act#fast"),
        lambda: endpoint.post("http://example.com/act"),
        lambda: endpoint.post("//example.com/act"),
        lambda: endpoint.post("/act", name=""),
        lambda: endpoint.post("/act", name=123),
        lambda: endpoint.post("/act", name="bad name"),
        lambda: endpoint.post("/act", name="bad%2Fname"),
        lambda: endpoint.post("/act", mode="batch"),
        lambda: endpoint.post("/act", description=123),
        lambda: endpoint.post("/act", tags="policy"),
        lambda: endpoint.post("/act", tags=(123,)),
        lambda: endpoint.post("/act", tags=("",)),
        lambda: endpoint.post("/act", tags=("bad tag",)),
        lambda: endpoint.post("/act", tags=("bad%2Ftag",)),
        lambda: EndpointSpec(method=" POST ", path="/act", name="act"),
        lambda: EndpointSpec(method="TRACE", path="/act", name="act"),
    ],
)
def test_endpoint_decorator_rejects_malformed_metadata(factory):
    with pytest.raises(ValueError):
        factory()


def test_custom_endpoint_discovery_rejects_duplicate_names_and_routes():
    class DuplicateNameTactic(EchoTactic):
        @endpoint.post("/one", name="same")
        async def one(self, input_value, *, context=None):
            return {"ok": True}

        @endpoint.post("/two", name="same")
        async def two(self, input_value, *, context=None):
            return {"ok": True}

    class DuplicateRouteTactic(EchoTactic):
        @endpoint.post("/same", name="one")
        async def one(self, input_value, *, context=None):
            return {"ok": True}

        @endpoint.post("/same", name="two")
        async def two(self, input_value, *, context=None):
            return {"ok": True}

    class EquivalentRouteTemplateTactic(EchoTactic):
        @endpoint.post("/items/{name}", name="by_name")
        async def by_name(self, input_value, *, context=None):
            return {"ok": True}

        @endpoint.post("/items/{id}", name="by_id")
        async def by_id(self, input_value, *, context=None):
            return {"ok": True}

    with pytest.raises(ValueError, match="Duplicate endpoint name"):
        custom_endpoints(DuplicateNameTactic())
    with pytest.raises(ValueError, match="Duplicate custom endpoint route"):
        custom_endpoints(DuplicateRouteTactic())
    with pytest.raises(ValueError, match="Duplicate custom endpoint route"):
        custom_endpoints(EquivalentRouteTemplateTactic())


def test_custom_endpoint_discovery_does_not_evaluate_properties():
    class PropertyTactic(EchoTactic):
        @property
        def expensive_metadata(self):
            raise AssertionError("property should not be evaluated")

        @endpoint.post("/property-act", name="property_act")
        async def property_act(self, input_value, *, context=None):
            return {"acted": input_value.text}

    endpoints = custom_endpoints(PropertyTactic())
    app = create_tactic_app(PropertyTactic())
    response = request(
        app,
        "POST",
        "/property-act",
        json={"input": {"text": "go"}},
    )

    endpoint_names = {spec.name for spec, _method in endpoints}
    assert "property_act" in endpoint_names
    assert response.status_code == 200
    assert response.json() == {"acted": "go"}


def test_custom_endpoint_discovery_does_not_call_custom_dir():
    class DirTactic(EchoTactic):
        def __dir__(self):
            raise AssertionError("__dir__ should not be called")

        @endpoint.post("/dir-act", name="dir_act")
        async def dir_act(self, input_value, *, context=None):
            return {"acted": input_value.text}

    endpoints = custom_endpoints(DirTactic())
    app = create_tactic_app(DirTactic())
    response = request(
        app,
        "POST",
        "/dir-act",
        json={"input": {"text": "go"}},
    )

    endpoint_names = {spec.name for spec, _method in endpoints}
    assert "dir_act" in endpoint_names
    assert response.status_code == 200
    assert response.json() == {"acted": "go"}


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


@pytest.mark.parametrize("value", ["false", "true", 0, 1])
def test_create_service_app_rejects_coerced_single_route_flag(value):
    with pytest.raises(TypeError, match="expose_single_tactic_routes"):
        create_service_app(
            [EchoTactic()],
            expose_single_tactic_routes=value,  # type: ignore[arg-type]
        )


def test_multi_tactic_service_lists_and_runs_named_tactics():
    app = create_service_app([EchoTactic(), StreamTactic()])

    listed = request(app, "GET", "/tactics")
    response = request(app, "POST", "/tactics/echo/run", json={"input": {"text": "hello"}})

    assert listed.status_code == 200
    assert {item["name"] for item in listed.json()} == {"echo", "streamer"}
    assert response.status_code == 200
    assert response.json()["output"] == {"text": "HELLO"}


def test_pydantic_ai_tactic_serves_through_fastapi_with_context_metadata():
    agent = ServicePydanticAgent()
    tactic = PydanticAITactic(
        agent,
        input_type=EchoInput,
        output_type=EchoOutput,
        input_mode="dict",
    )
    app = create_tactic_app(tactic)

    info = request(app, "GET", "/info")
    response = request(
        app,
        "POST",
        "/run",
        json={
            "input": {"text": "hello"},
            "context": {
                "request_id": "req-pydantic-service",
                "trace_id": "trace-pydantic-service",
                "metadata": {"caller": "fastapi-test"},
            },
        },
    )

    assert info.status_code == 200
    assert info.json()["name"] == "pydantic-service"
    assert info.json()["runtime_kind"] == "pydantic-ai"
    assert response.status_code == 200
    assert response.json() == {
        "output": {"text": "HELLO:trace-pydantic-service"},
        "request_id": "req-pydantic-service",
        "tactic": "pydantic-service",
    }
    assert agent.seen_task == {"text": "hello"}
    assert agent.seen_metadata["caller"] == "fastapi-test"
    assert agent.seen_metadata["lllm_trace_id"] == "trace-pydantic-service"
    assert agent.seen_metadata["lllm_endpoint"] == "run"


def test_service_info_filters_secret_examples_and_metadata():
    tactic = EchoTactic(
        examples=[
            {
                "input": {
                    "text": "hello",
                    "headers": {
                        "authorization": "Bearer raw-example-auth",
                        "x-api-key": "raw-example-key",
                        "x-policy": "safe-policy",
                    },
                },
                "output": {"password": "raw-example-password", "text": "HELLO"},
            }
        ],
        metadata={
            "api_key_ref": "credentials/openai",
            "headers": {
                "authorization": "Bearer raw-metadata-auth",
                "x-policy": "safe-metadata-policy",
            },
        },
    )
    app = create_tactic_app(tactic)

    info = request(app, "GET", "/info").json()
    listed = request(app, "GET", "/tactics").json()

    for payload in (info, listed[0]):
        text = json.dumps(payload, sort_keys=True)
        assert "raw-example-auth" not in text
        assert "raw-example-key" not in text
        assert "raw-example-password" not in text
        assert "raw-metadata-auth" not in text
        assert "safe-policy" in text
        assert "safe-metadata-policy" in text
        assert payload["metadata"]["api_key_ref"] == "credentials/openai"


def test_service_rejects_path_control_tactic_route_names():
    bad_names = (
        "",
        "   ",
        ".",
        "..",
        "bad/name",
        r"bad\name",
        "bad:name",
        "bad name",
        "bad%2Fname",
        None,
        123,
    )

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


def test_custom_endpoint_routes_cannot_shadow_service_routes():
    class ShadowRunTactic(EchoTactic):
        @endpoint.post("/run", name="shadow_run")
        async def shadow_run(self, input_value, *, context=None):
            return {"ok": True}

    class ShadowNamedRunTactic(EchoTactic):
        @endpoint.post("/tactics/echo/run", name="shadow_named_run")
        async def shadow_named_run(self, input_value, *, context=None):
            return {"ok": True}

    class ShadowNamedRouteTemplateTactic(EchoTactic):
        @endpoint.post("/tactics/{tactic}/run", name="shadow_named_template")
        async def shadow_named_template(self, input_value, *, context=None):
            return {"ok": True}

    with pytest.raises(ValueError, match="reserved LLLM service route"):
        create_tactic_app(ShadowRunTactic())
    with pytest.raises(ValueError, match="reserved LLLM service route"):
        create_service_app([ShadowNamedRunTactic()])
    with pytest.raises(ValueError, match="reserved LLLM service route"):
        create_service_app([ShadowNamedRouteTemplateTactic()])


def test_custom_endpoint_routes_are_unique_across_service():
    class OtherActTactic(Tactic[EchoInput, EchoOutput]):
        name = "other"
        input_type = EchoInput
        output_type = EchoOutput

        def _run(self, input_value, *, context=None):
            return EchoOutput(text=input_value.text)

        @endpoint.post("/act")
        async def act(self, input_value, *, context=None):
            return {"acted": input_value.text}

    with pytest.raises(ValueError, match="Duplicate custom endpoint route"):
        create_service_app([EchoTactic(), OtherActTactic()])

    class ByNameTactic(Tactic[EchoInput, EchoOutput]):
        name = "by-name"
        input_type = EchoInput
        output_type = EchoOutput

        def _run(self, input_value, *, context=None):
            return EchoOutput(text=input_value.text)

        @endpoint.post("/custom/{name}")
        async def custom_by_name(self, input_value, *, context=None):
            return {"name": input_value.text}

    class ByIdTactic(Tactic[EchoInput, EchoOutput]):
        name = "by-id"
        input_type = EchoInput
        output_type = EchoOutput

        def _run(self, input_value, *, context=None):
            return EchoOutput(text=input_value.text)

        @endpoint.post("/custom/{id}")
        async def custom_by_id(self, input_value, *, context=None):
            return {"id": input_value.text}

    with pytest.raises(ValueError, match="Duplicate custom endpoint route"):
        create_service_app([ByNameTactic(), ByIdTactic()])


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
