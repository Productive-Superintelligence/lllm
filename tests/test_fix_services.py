"""Regression tests for service fixes S1 (RemoteTactic default names), S2
(custom GET/DELETE endpoints without a body) and S3 (tactic schemas in OpenAPI)."""

import asyncio
import datetime
import decimal
import enum
import json
import math
import threading
import uuid
from typing import Annotated, Literal, Union

import httpx
import pytest
from pydantic import BaseModel, ConfigDict, Field

from lllm import Tactic, TacticInfo, endpoint
from lllm.services import RemoteTactic, create_service_app, create_tactic_app
from lllm.services.endpoints import EndpointSpec, endpoint_requires_body


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


# S2: custom GET/DELETE requests without a body do not validate against the
# input type; a body, when sent, is still read and validated.


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

    @endpoint.delete("/policy")
    async def delete_policy(self, input_value, *, context=None):
        return await self.arun(input_value, context=context)

    @endpoint.post("/act")
    async def act(self, input_value, *, context=None):
        return {"type": type(input_value).__name__, "text": input_value.text}

    @endpoint.patch("/act")
    async def patch_act(self, input_value, *, context=None):
        return {"type": type(input_value).__name__, "text": input_value.text}


@pytest.mark.parametrize(
    ("method", "requires_body"),
    [("GET", False), ("DELETE", False), ("POST", True), ("PUT", True), ("PATCH", True)],
)
def test_endpoint_requires_body_follows_http_method(method, requires_body):
    spec = EndpointSpec(method=method, path="/route", name="route")

    assert endpoint_requires_body(spec) is requires_body


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


def test_delete_endpoint_without_body_skips_input_validation():
    response = call(create_tactic_app(ItemTactic()), "DELETE", "/items/7")

    assert response.status_code == 200
    assert response.json() == {"input": None}


def test_delete_endpoint_reads_and_validates_body_when_present():
    app = create_tactic_app(ItemTactic())

    envelope = call(app, "DELETE", "/policy", json={"input": {"text": "forward"}})
    raw = call(app, "DELETE", "/policy", json={"text": "forward"})
    query = call(app, "DELETE", "/policy", params={"text": "forward"})
    invalid = call(
        app,
        "DELETE",
        "/policy",
        json={"input": {"nope": 1}, "context": {"request_id": "req-delete"}},
    )

    for response in (envelope, raw, query):
        assert response.status_code == 200
        assert response.json() == {"text": "forward"}
    assert invalid.status_code == 400
    assert invalid.json()["detail"]["error"]["type"] == "SchemaError"
    assert invalid.json()["detail"]["error"]["request_id"] == "req-delete"


def test_get_endpoint_reads_and_validates_body_when_present():
    app = create_tactic_app(ItemTactic())

    valid = call(app, "GET", "/status", json={"input": {"text": "go"}}, params={"q": "x"})
    invalid = call(app, "GET", "/status", json={"nope": 1})

    assert valid.status_code == 200
    assert valid.json() == {"input": {"text": "go"}, "endpoint": "status"}
    assert invalid.status_code == 400
    assert invalid.json()["detail"]["error"]["type"] == "SchemaError"


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
    """Yield ``$ref`` values and discriminator mapping targets."""

    if isinstance(value, dict):
        for key, item in value.items():
            if key == "$ref" and isinstance(item, str):
                yield item
            else:
                yield from _refs(item)
        discriminator = value.get("discriminator")
        if isinstance(discriminator, dict):
            for target in (discriminator.get("mapping") or {}).values():
                if isinstance(target, str):
                    yield target
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
    class Address(BaseModel):
        """Not the Address used by RichInput, but defined under the same name."""

        zip: str

    class OtherOutput(BaseModel):
        address: Address
        more: list[Address] = []

    class ConflictTactic(RichTactic):
        output_type = OtherOutput

    # A real clash: both schemas name a different model "Address" in $defs.
    assert "Address" in RichInput.model_json_schema()["$defs"]
    assert set(OtherOutput.model_json_schema()["$defs"]) == {"Address"}

    document = create_tactic_app(ConflictTactic()).openapi()

    input_address = _request_schema(document)["properties"]["input"]["properties"]["address"]
    output = _response_schema(document)["properties"]["output"]["properties"]
    assert input_address["anyOf"][0] == {"$ref": "#/components/schemas/Address"}
    assert output["address"] == {"$ref": "#/components/schemas/Address2"}
    assert output["more"]["items"] == output["address"]
    assert "city" in _resolve(document, "#/components/schemas/Address")["properties"]
    assert _resolve(document, "#/components/schemas/Address2")["properties"] == {
        "zip": {"title": "Zip", "type": "string"}
    }
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

    # Force regeneration: FastAPI 0.137+ also does it when routes change,
    # older versions keep serving the cached document.
    app.openapi_schema = None
    second = app.openapi()

    assert app.openapi() is second
    assert second is not first
    assert _request_schema(second)["properties"]["input"]["title"] == "RichInput"
    assert "Address2" not in second["components"]["schemas"]
    assert "ValidationError3" not in second["components"]["schemas"]
    _assert_refs_resolve(second)


class Cat(BaseModel):
    kind: Literal["cat"] = "cat"


class Dog(BaseModel):
    kind: Literal["dog"] = "dog"


class PetInput(BaseModel):
    pet: Annotated[Union[Cat, Dog], Field(discriminator="kind")]


class PetTactic(Tactic[PetInput, PetInput]):
    name = "pets"
    input_type = PetInput
    output_type = PetInput

    def _run(self, input_value, *, context=None):
        return input_value


def test_openapi_rewrites_discriminator_mapping_targets():
    document = create_tactic_app(PetTactic()).openapi()

    for schema in (
        _request_schema(document)["properties"]["input"],
        _request_schema(document, "/stream")["properties"]["input"],
        _response_schema(document)["properties"]["output"],
    ):
        assert schema["properties"]["pet"]["discriminator"]["mapping"] == {
            "cat": "#/components/schemas/Cat",
            "dog": "#/components/schemas/Dog",
        }
    _assert_refs_resolve(document)


def test_openapi_discriminator_mapping_follows_renamed_member():
    class ValidationError(BaseModel):
        """Shares its name with FastAPI's built-in component."""

        kind: Literal["invalid"] = "invalid"
        reason: str

    class Accepted(BaseModel):
        kind: Literal["ok"] = "ok"

    class Verdict(BaseModel):
        verdict: Annotated[Union[ValidationError, Accepted], Field(discriminator="kind")]

    class JudgeTactic(Tactic[Verdict, Verdict]):
        name = "judge"
        input_type = Verdict
        output_type = Verdict

        def _run(self, input_value, *, context=None):
            return input_value

    document = create_tactic_app(JudgeTactic()).openapi()
    verdict = _request_schema(document)["properties"]["input"]["properties"]["verdict"]
    mapping = verdict["discriminator"]["mapping"]

    assert mapping == {
        "invalid": "#/components/schemas/ValidationError2",
        "ok": "#/components/schemas/Accepted",
    }
    assert [choice["$ref"] for choice in verdict["oneOf"]] == list(mapping.values())
    assert "reason" in _resolve(document, mapping["invalid"])["properties"]
    _assert_refs_resolve(document)


def test_openapi_is_served_while_tactic_info_fails(caplog):
    failing = True

    class FlakyInfoTactic(RichTactic):
        def info(self):
            if failing:
                raise RuntimeError("info backend unavailable")
            return super().info()

        @endpoint.post("/act")
        async def act(self, input_value, *, context=None):
            return input_value

    app = create_service_app([FlakyInfoTactic()], expose_single_tactic_routes=True)

    with caplog.at_level("WARNING", logger="lllm.services.fastapi"):
        served = call(app, "GET", "/openapi.json")
    failing = False
    recovered = call(app, "GET", "/openapi.json")

    assert served.status_code == 200
    assert "/act" in served.json()["paths"]
    assert "requestBody" not in served.json()["paths"]["/run"]["post"]
    assert "Could not add the schemas of tactic 'rich'" in caplog.text
    assert recovered.status_code == 200
    document = recovered.json()
    assert _request_schema(document)["properties"]["input"]["title"] == "RichInput"
    assert "Address2" not in document["components"]["schemas"]
    _assert_refs_resolve(document)


def test_openapi_warns_once_while_adding_schemas_keeps_failing(caplog):
    failing = True

    class FlakyInfoTactic(RichTactic):
        def info(self):
            if failing:
                raise RuntimeError("info backend unavailable")
            return super().info()

    app = create_service_app([FlakyInfoTactic()], expose_single_tactic_routes=True)

    def records():
        return [record for record in caplog.records if record.name == "lllm.services.fastapi"]

    with caplog.at_level("DEBUG", logger="lllm.services.fastapi"):
        for _ in range(3):
            assert call(app, "GET", "/openapi.json").status_code == 200
        streak = [record.levelname for record in records()]
        failing = False
        recovered = call(app, "GET", "/openapi.json").json()
        failing = True
        app.openapi_schema = None  # A regenerated document starts a new streak.
        assert call(app, "GET", "/openapi.json").status_code == 200

    assert streak == ["WARNING", "DEBUG", "DEBUG"]
    assert "requestBody" in recovered["paths"]["/run"]["post"]
    assert [record.levelname for record in records()] == streak + ["WARNING"]
    assert all(record.exc_info for record in records())


def test_openapi_is_served_when_tactic_schema_cannot_be_exported():
    class Opaque:
        pass

    class OpaqueInput(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)

        thing: Opaque

    class OpaqueTactic(Tactic):
        name = "opaque"
        input_type = OpaqueInput

        def _run(self, input_value, *, context=None):
            return None

    app = create_service_app([OpaqueTactic()], expose_single_tactic_routes=True)

    response = call(app, "GET", "/openapi.json")

    assert response.status_code == 200
    assert "requestBody" not in response.json()["paths"]["/run"]["post"]


class Budget(BaseModel):
    text: str
    max_cost: float = math.inf
    min_cost: float = -math.inf
    ratio: float = math.nan


class Order(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "id": uuid.UUID(int=7),
                    "placed_at": datetime.datetime(2024, 1, 1),
                    "total": decimal.Decimal("1.5"),
                }
            ]
        }
    )

    id: uuid.UUID


@pytest.mark.parametrize("model", [Budget, Order])
@pytest.mark.parametrize("role", ["input", "output"])
def test_served_openapi_encodes_tactic_schemas_like_info(model, role):
    class EncodedTactic(Tactic):
        name = "encoded"
        input_type = model if role == "input" else None
        output_type = model if role == "output" else None

        def _run(self, input_value, *, context=None):
            return input_value

    app = create_tactic_app(EncodedTactic())

    served = call(app, "GET", "/openapi.json")
    info = call(app, "GET", "/info").json()

    assert served.status_code == 200
    document = served.json()
    if role == "input":
        documented = _request_schema(document)["properties"]["input"]
    else:
        documented = _response_schema(document)["properties"]["output"]
    assert documented == info[f"{role}_schema"]
    _assert_refs_resolve(document)


def _non_finite_info_tactic():
    class NonFiniteInfo(TacticInfo):
        model_config = ConfigDict(ser_json_inf_nan="constants")

    class NonFiniteInfoTactic(Tactic):
        name = "constants"
        input_type = Budget

        def _run(self, input_value, *, context=None):
            return None

        def info(self):
            return NonFiniteInfo(**super().info().model_dump())

    return NonFiniteInfoTactic()


def _unpaired_surrogate_tactic():
    class SurrogateInput(BaseModel):
        text: str = Field(description="unpaired \ud800")

    class SurrogateTactic(Tactic):
        name = "surrogate"
        input_type = SurrogateInput

        def _run(self, input_value, *, context=None):
            return None

    return SurrogateTactic()


@pytest.mark.parametrize("make_tactic", [_non_finite_info_tactic, _unpaired_surrogate_tactic])
def test_openapi_is_served_without_schemas_it_cannot_render(make_tactic, caplog):
    app = create_tactic_app(make_tactic())

    with caplog.at_level("WARNING", logger="lllm.services.fastapi"):
        response = call(app, "GET", "/openapi.json")

    assert response.status_code == 200
    assert "requestBody" not in response.json()["paths"]["/run"]["post"]
    assert "Could not add the schemas of tactic" in caplog.text


def test_openapi_documents_schemas_once_for_concurrent_first_callers():
    armed = threading.Event()
    patching = threading.Event()
    release = threading.Event()
    patches = []

    class SlowInfoTactic(RichTactic):
        def info(self):
            if armed.is_set():
                patches.append(threading.current_thread().name)
                patching.set()
                release.wait(timeout=5)
            return super().info()

    app = create_tactic_app(SlowInfoTactic())
    armed.set()
    documents = []
    first = threading.Thread(target=lambda: documents.append(app.openapi()))
    second = threading.Thread(target=lambda: documents.append(app.openapi()))
    first.start()
    assert patching.wait(timeout=5)
    second.start()
    second.join(timeout=0.2)  # Without serialization it starts patching too.
    release.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert len(patches) == 1
    assert len(documents) == 2 and documents[0] is documents[1]
    assert "Address2" not in documents[0]["components"]["schemas"]
    _assert_refs_resolve(documents[0])


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
