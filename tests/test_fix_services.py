"""Regression tests for service fixes S1 (RemoteTactic default names), S2
(bodiless custom endpoints) and S3 (tactic schemas in OpenAPI)."""

import asyncio
import enum
import json

import httpx
import pytest
from pydantic import BaseModel

from lllm import Tactic, endpoint
from lllm.services import RemoteTactic, create_service_app, create_tactic_app
from lllm.services.endpoints import EndpointSpec, endpoint_has_body


def call(app, method, path, **kwargs):
    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.request(method, path, **kwargs)
            await response.aread()
            return response

    return asyncio.run(run())


# S1: RemoteTactic(url) derives a valid default name.


@pytest.mark.parametrize(
    ("url", "name"),
    [
        ("http://127.0.0.1:8000", "127.0.0.1-8000"),
        ("http://127.0.0.1:8000/", "127.0.0.1-8000"),
        ("http://127.0.0.1:8000/run", "127.0.0.1-8000"),
        ("http://127.0.0.1:8000/stream", "127.0.0.1-8000"),
        ("http://127.0.0.1:8000/info", "127.0.0.1-8000"),
        ("http://localhost:8000", "localhost-8000"),
        ("http://[::1]:8000", "[--1]-8000"),
        ("http://localhost:8000/tactics/echo/run", "echo"),
        ("http://testserver/run", "testserver"),
        ("https://example.com", "example.com"),
    ],
)
def test_remote_tactic_derives_valid_default_name(url, name):
    remote = RemoteTactic(url)

    assert remote.tactic_name == name
    assert remote.info().name == name


def test_remote_tactic_with_default_name_runs_and_mounts():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, json={"output": "HELLO"})

    remote = RemoteTactic(
        "http://127.0.0.1:8000",
        transport=httpx.MockTransport(handler),
    )

    assert remote.run("hello") == "HELLO"
    assert seen == ["http://127.0.0.1:8000/run"]
    app = create_service_app([remote])
    assert app.state.lllm_tactics == {"127.0.0.1-8000": remote}


# S2: bodiless custom endpoints do not validate against the input type.


class ItemInput(BaseModel):
    text: str


class ItemOutput(BaseModel):
    text: str


class ItemTactic(Tactic[ItemInput, ItemOutput]):
    name = "items"
    input_type = ItemInput
    output_type = ItemOutput

    def _run(self, input_value, *, context=None):
        return ItemOutput(text=input_value.text)

    @endpoint.get("/status")
    async def status(self, input_value, *, context=None):
        return {"input": input_value, "endpoint": context.endpoint}

    @endpoint.get("/feed", mode="events")
    def feed(self, input_value, *, context=None):
        yield {"input": input_value}

    @endpoint.delete("/items/{item_id}")
    async def remove(self, input_value, *, context=None):
        return {"input": input_value}

    @endpoint.post("/act")
    async def act(self, input_value, *, context=None):
        return {"type": type(input_value).__name__, "text": input_value.text}

    @endpoint.patch("/act")
    async def patch_act(self, input_value, *, context=None):
        return {"type": type(input_value).__name__, "text": input_value.text}


@pytest.mark.parametrize(
    ("method", "has_body"),
    [("GET", False), ("DELETE", False), ("POST", True), ("PUT", True), ("PATCH", True)],
)
def test_endpoint_has_body_follows_http_method(method, has_body):
    spec = EndpointSpec(method=method, path="/route", name="route")

    assert endpoint_has_body(spec) is has_body


def test_get_endpoint_receives_none_without_query_params():
    response = call(create_tactic_app(ItemTactic()), "GET", "/status")

    assert response.status_code == 200
    assert response.json() == {"input": None, "endpoint": "status"}


def test_get_endpoint_receives_query_params_unvalidated():
    response = call(
        create_tactic_app(ItemTactic()),
        "GET",
        "/status",
        params=[("verbose", "1"), ("tag", "a"), ("tag", "b")],
    )

    assert response.status_code == 200
    assert response.json() == {
        "input": {"verbose": "1", "tag": ["a", "b"]},
        "endpoint": "status",
    }


def test_get_endpoint_works_in_multi_tactic_app():
    app = create_service_app([ItemTactic()])

    response = call(app, "GET", "/status", params={"q": "x"})

    assert response.status_code == 200
    assert response.json()["input"] == {"q": "x"}


def test_get_events_endpoint_streams_query_input():
    response = call(create_tactic_app(ItemTactic()), "GET", "/feed", params={"since": "3"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    assert events == [{"input": {"since": "3"}}]


def test_delete_endpoint_skips_input_validation():
    response = call(create_tactic_app(ItemTactic()), "DELETE", "/items/7")

    assert response.status_code == 200
    assert response.json() == {"input": None}


@pytest.mark.parametrize("method", ["POST", "PATCH"])
def test_body_endpoints_still_validate_input(method):
    app = create_tactic_app(ItemTactic())

    valid = call(app, method, "/act", json={"input": {"text": "go"}})
    missing = call(app, method, "/act")
    invalid = call(app, method, "/act", json={"nope": 1})

    assert valid.status_code == 200
    assert valid.json() == {"type": "ItemInput", "text": "go"}
    for response in (missing, invalid):
        assert response.status_code == 400
        assert response.json()["detail"]["error"]["type"] == "SchemaError"


# S3: OpenAPI documents the tactic input/output schemas on /run.


class Color(enum.Enum):
    RED = "red"
    BLUE = "blue"


class Address(BaseModel):
    city: str
    color: Color = Color.RED


class Node(BaseModel):
    value: int
    children: list["Node"] = []


class ValidationError(BaseModel):
    """Shares its name with FastAPI's built-in component."""

    reason: str


class RichInput(BaseModel):
    text: str
    address: Address | None = None
    tree: Node | None = None
    problem: ValidationError | None = None


class RichOutput(BaseModel):
    text: str
    address: Address | None = None


class RichTactic(Tactic[RichInput, RichOutput]):
    name = "rich"
    input_type = RichInput
    output_type = RichOutput

    def _run(self, input_value, *, context=None):
        return RichOutput(text=input_value.text, address=input_value.address)


def _resolve(document, ref):
    assert ref.startswith("#/"), ref
    node = document
    for part in ref[2:].split("/"):
        node = node[part.replace("~1", "/").replace("~0", "~")]
    return node


def _refs(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "$ref" and isinstance(item, str):
                yield item
            else:
                yield from _refs(item)
    elif isinstance(value, list):
        for item in value:
            yield from _refs(item)


def _assert_refs_resolve(document):
    for ref in _refs(document):
        _resolve(document, ref)
    assert "$defs" not in json.dumps(document)


def _request_schema(document, path="/run"):
    body = document["paths"][path]["post"]["requestBody"]
    return body["content"]["application/json"]["schema"]


def _response_schema(document):
    response = document["paths"]["/run"]["post"]["responses"]["200"]
    return response["content"]["application/json"]["schema"]


def test_openapi_documents_run_input_and_output_schemas():
    document = create_tactic_app(RichTactic()).openapi()

    request_schema = _request_schema(document)
    response_schema = _response_schema(document)
    assert document["paths"]["/run"]["post"]["requestBody"]["required"] is True
    assert request_schema["properties"]["input"]["title"] == "RichInput"
    assert "context" in request_schema["properties"]
    assert response_schema["properties"]["output"]["title"] == "RichOutput"
    assert response_schema["required"] == ["request_id", "tactic"]
    assert _request_schema(document, "/stream")["properties"]["input"]["title"] == "RichInput"
    _assert_refs_resolve(document)


def test_openapi_hoists_nested_defs_without_clobbering_components():
    document = create_tactic_app(RichTactic()).openapi()
    schemas = document["components"]["schemas"]
    input_schema = _request_schema(document)["properties"]["input"]

    assert "loc" in schemas["ValidationError"]["properties"]
    problem_ref = input_schema["properties"]["problem"]["anyOf"][0]["$ref"]
    assert _resolve(document, problem_ref)["properties"] == {
        "reason": {"title": "Reason", "type": "string"}
    }
    assert "Address" in schemas and "Address2" not in schemas
    assert schemas["Node"]["properties"]["children"]["items"] == {
        "$ref": "#/components/schemas/Node"
    }
    assert schemas["Color"]["enum"] == ["red", "blue"]


def test_openapi_keeps_same_named_input_and_output_models_apart():
    class OtherAddress(BaseModel):
        zip: str

    OtherAddress.__name__ = "Address"

    class OtherOutput(BaseModel):
        address: OtherAddress

    class ConflictTactic(RichTactic):
        output_type = OtherOutput

    document = create_tactic_app(ConflictTactic()).openapi()

    input_ref = _request_schema(document)["properties"]["input"]["properties"]["address"]
    output_ref = _response_schema(document)["properties"]["output"]["properties"]["address"]
    assert "city" in _resolve(document, input_ref["anyOf"][0]["$ref"])["properties"]
    assert "zip" in _resolve(document, output_ref["$ref"])["properties"]
    _assert_refs_resolve(document)


def test_openapi_uses_schemas_advertised_by_info():
    class CustomInfoTactic(RichTactic):
        def info(self):
            info = super().info()
            info.output_schema = {"type": "string", "title": "Advertised"}
            return info

    app = create_tactic_app(CustomInfoTactic())

    assert _response_schema(app.openapi())["properties"]["output"] == {
        "type": "string",
        "title": "Advertised",
    }


def test_openapi_documents_untyped_tactic_input_as_any():
    class UntypedTactic(Tactic):
        name = "untyped"

        def _run(self, input_value, *, context=None):
            return input_value

    document = create_tactic_app(UntypedTactic()).openapi()

    assert _request_schema(document)["properties"]["input"] == {}
    assert _response_schema(document)["properties"]["output"] == {}


def test_openapi_schemas_survive_regeneration_without_duplicates():
    app = create_tactic_app(RichTactic())
    first = app.openapi()

    @app.get("/extra")
    async def extra():
        return {}

    second = app.openapi()

    assert app.openapi() is second
    assert second is not first
    assert _request_schema(second)["properties"]["input"]["title"] == "RichInput"
    assert "Address2" not in second["components"]["schemas"]
    assert "ValidationError3" not in second["components"]["schemas"]
    _assert_refs_resolve(second)


def test_served_openapi_is_typed_and_run_behaviour_is_unchanged():
    app = create_tactic_app(RichTactic())

    served = call(app, "GET", "/openapi.json").json()
    envelope = call(app, "POST", "/run", json={"input": {"text": "hi"}})
    raw = call(app, "POST", "/run", json={"text": "raw"})
    invalid = call(app, "POST", "/run", json={"input": {"nope": 1}})

    assert _request_schema(served)["properties"]["input"]["title"] == "RichInput"
    assert envelope.status_code == 200
    assert envelope.json()["output"]["text"] == "hi"
    assert raw.status_code == 200
    assert raw.json()["output"]["text"] == "raw"
    assert invalid.status_code == 400
    assert invalid.json()["detail"]["error"]["type"] == "SchemaError"
