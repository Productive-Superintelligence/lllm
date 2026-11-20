"""Regression tests for serving, the CLI, RemoteTactic, and the hub (fix3)."""

from __future__ import annotations

import asyncio
import functools
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Literal

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel, ValidationError

from lllm import RemoteTactic, RemoteTacticError, SchemaError, endpoint
from lllm import Tactic as NativeTactic
from lllm.cli import load_tactic_entrypoint, main
from lllm.protocol import OutputSchemaError
from lllm.protocol import Tactic as ProtocolTactic
from lllm.services import create_service_app, create_tactic_app, status_code_for_error


@pytest.fixture(autouse=True)
def _forget_fix3_modules(monkeypatch):
    """Drop the modules and paths this file's tests import or serve."""

    monkeypatch.setattr(sys, "path", list(sys.path))
    yield
    for name in list(sys.modules):
        if name.split(".", 1)[0].startswith("fix3_"):
            del sys.modules[name]


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.lstrip(), encoding="utf-8")
    return path


# 1: `lllm serve module:attr` and `lllm inspect module:attr` serve an app, or
# a factory's app, as the app instead of wrapping it as a callable tactic.

APP_MODULE = """
from fastapi import FastAPI
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

CALLS = []

app = FastAPI(title="mine")


@app.get("/hello")
def hello():
    return {"hello": "app"}


def create_app():
    CALLS.append("create_app")
    return app


async def _ping(request):
    return JSONResponse({"pong": True})


starlette_app = Starlette(routes=[Route("/ping", _ping)])


def shout(text: str) -> str:
    return text.upper()


def build_number():
    CALLS.append("build_number")
    return 42
"""


@pytest.fixture
def app_module(tmp_path, monkeypatch):
    _write(tmp_path / "fix3_apps.py", APP_MODULE)
    monkeypatch.syspath_prepend(str(tmp_path))
    return "fix3_apps"


@pytest.fixture
def served(monkeypatch):
    """Capture what `lllm serve` hands to Uvicorn instead of starting it."""

    calls = []
    monkeypatch.setattr(
        "uvicorn.run",
        lambda app, **options: calls.append((app, options)),
    )
    return calls


@pytest.mark.parametrize("attr", ["app", "create_app"])
def test_serve_runs_an_app_entrypoint_as_the_app(app_module, served, attr):
    assert main(["serve", f"{app_module}:{attr}", "--port", "8123"]) == 0

    import fix3_apps

    [(app, options)] = served
    assert app is fix3_apps.app
    assert options["port"] == 8123
    assert TestClient(app).get("/hello").json() == {"hello": "app"}
    assert fix3_apps.CALLS == ([] if attr == "app" else ["create_app"])


def test_serve_runs_a_starlette_app_entrypoint_as_the_app(app_module, served):
    assert main(["serve", f"{app_module}:starlette_app"]) == 0

    import fix3_apps

    [(app, options)] = served
    assert app is fix3_apps.starlette_app
    assert options["port"] == 8000


def test_serve_still_wraps_a_handler_entrypoint_as_a_tactic(app_module, served):
    assert main(["serve", f"{app_module}:shout"]) == 0

    [(app, _options)] = served
    client = TestClient(app)
    assert client.post("/run", json={"input": "hi"}).json()["output"] == "HI"
    assert client.get("/info").json()["name"] == "shout"


@pytest.mark.parametrize("attr", ["app", "create_app"])
def test_inspect_reports_an_app_entrypoint_as_an_app(app_module, capsys, attr):
    assert main(["inspect", f"{app_module}:{attr}"]) == 0

    out = capsys.readouterr().out
    assert out.splitlines()[0] == f"{app_module}:{attr} (FastAPI app)"
    assert "GET /hello" in out
    assert "callable" not in out

    assert main(["inspect", f"{app_module}:{attr}", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["kind"] == "app"
    assert payload["app"] == "FastAPI"
    assert payload["label"] == f"{app_module}:{attr}"
    assert payload["port"] == 8000
    assert "GET /hello" in payload["routes"]


def test_inspect_still_reports_a_tactic_entrypoint(app_module, capsys):
    assert main(["inspect", f"{app_module}:shout"]) == 0

    assert capsys.readouterr().out.splitlines()[0] == "shout (python)"


@pytest.mark.parametrize("attr", ["app", "create_app", "starlette_app"])
def test_load_tactic_entrypoint_refuses_an_app(app_module, attr):
    with pytest.raises(TypeError, match="app, not a tactic"):
        load_tactic_entrypoint(f"{app_module}:{attr}")


@pytest.mark.parametrize("command", ["serve", "inspect"])
def test_a_factory_that_returns_nothing_usable_is_an_error(
    app_module, served, capsys, command
):
    with pytest.raises(SystemExit) as caught:
        main([command, f"{app_module}:build_number"])

    assert caught.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == "" and served == []
    assert "fix3_apps:build_number did not resolve" in captured.err
    assert "the factory returned int" in captured.err  # what it got
    assert "a FastAPI or Starlette app" in captured.err  # what it accepts
    assert "Traceback" not in captured.err


def test_load_tactic_entrypoint_explains_a_factory_that_returns_nothing_usable(
    app_module,
):
    with pytest.raises(TypeError, match="the factory returned int; it must return"):
        load_tactic_entrypoint(f"{app_module}:build_number")


@pytest.mark.parametrize("entry", ["app", "create_app"])
def test_a_tactic_entry_that_names_an_app_is_a_launch_error(tmp_path, entry):
    from lllm.launch import LaunchError, load_launch_app

    package = _web_package(tmp_path, entry)

    with pytest.raises(LaunchError) as caught:
        load_launch_app(package, resource="tactics.web")
    assert (
        f"tactic web entry fix3_web.app:{entry} is or returns a FastAPI or "
        "Starlette app, not a tactic; serve an app from a [services.NAME] entry"
    ) in str(caught.value)

    with pytest.raises(LaunchError) as junk:
        load_launch_app(package, resource="tactics.number")
    assert str(junk.value).endswith(
        "the factory returned int; it must return a tactic or Tactic subclass, "
        "an object with run(), or a callable"
    )


def _web_package(tmp_path: Path, entry: str) -> Path:
    package = tmp_path / "web"
    _write(
        package / "lllm.toml",
        f"""
[package]
org = "demo"
name = "web"

[tactics.web]
entry = "fix3_web.app:{entry}"

[tactics.number]
entry = "fix3_web.app:build_number"
""",
    )
    _write(package / "fix3_web" / "__init__.py", "")
    _write(package / "fix3_web" / "app.py", APP_MODULE)
    return package


def test_validate_refuses_a_tactic_entry_that_is_an_app(tmp_path):
    from lllm.hub import LocalHub, PublishValidationError, validate_package

    package = _web_package(tmp_path, "app")

    report = validate_package(package)

    assert not report.ok
    [issue] = [issue for issue in report.issues if issue.code == "tactic_entry_is_app"]
    assert issue.level == "error"
    assert issue.resource == "lllm://demo/web/tactics/web"
    assert "fix3_web.app:app is a FastAPI or Starlette app" in issue.message
    assert "[services.NAME]" in issue.message
    with pytest.raises(PublishValidationError):
        LocalHub(tmp_path / "hub").publish(package)


def test_validate_does_not_call_a_tactic_entry_factory(tmp_path):
    from lllm.hub import validate_package

    package = _web_package(tmp_path, "create_app")
    _write(
        package / "fix3_web" / "app.py",
        """
from pathlib import Path

from fastapi import FastAPI


def create_app():
    Path(__file__).with_name("called").write_text("yes")
    return FastAPI()


def build_number():
    return 42
""",
    )

    report = validate_package(package)

    assert report.ok
    assert not (package / "fix3_web" / "called").exists()


# 2: a result that fails the tactic's output schema is a bug in the tactic,
# so a service answers it with 500, not the 400 of bad client input.


class Route(BaseModel):
    queue: Literal["billing", "bugs"]


class RouteTactic(ProtocolTactic):
    name = "route"
    input_type = str
    output_type = Route

    def _run(self, text, *, context=None, **kwargs):
        return {"queue": "sales" if "sales" in text else "billing"}


class NativeRoute(NativeTactic):
    name = "native_route"
    input_type = str
    output_model = Route

    def call(self, text: str) -> dict:
        return {"queue": "sales" if "sales" in text else "billing"}


@pytest.mark.parametrize("tactic", [RouteTactic, NativeRoute])
@pytest.mark.parametrize("path", ["/run", "/tactics/{name}/run"])
def test_an_output_validation_failure_is_a_500(tactic, path):
    tactic = tactic()
    client = TestClient(create_tactic_app(tactic))
    url = path.format(name=tactic.tactic_name)
    body = {"input": "a sales lead", "context": {"request_id": "req-out"}}

    response = client.post(url, json=body)

    assert response.status_code == 500
    error = response.json()["detail"]["error"]
    assert error["type"] == "OutputSchemaError"
    assert error["message"].startswith(
        f"The output of tactic '{tactic.tactic_name}' failed validation: "
        "Value does not match schema Route"
    )
    assert error["tactic"] == tactic.tactic_name
    assert error["endpoint"] == "run"
    assert error["request_id"] == "req-out"


def test_bad_input_is_still_a_400():
    client = TestClient(create_tactic_app(RouteTactic()))

    response = client.post("/run", json={"input": 42})

    assert response.status_code == 400
    assert response.json()["detail"]["error"]["type"] == "SchemaError"
    assert client.post("/run", json={"input": "billing"}).status_code == 200


def test_an_output_schema_error_is_a_schema_error_caused_by_pydantic():
    with pytest.raises(SchemaError) as caught:
        RouteTactic().run("a sales lead")

    assert isinstance(caught.value, OutputSchemaError)
    assert isinstance(caught.value.__cause__, ValidationError)
    assert caught.value.__cause__.title == "Route"
    with pytest.raises(SchemaError) as bad_input:
        RouteTactic().run(42)
    assert not isinstance(bad_input.value, OutputSchemaError)


def test_status_code_for_error_maps_output_failures_to_500():
    assert status_code_for_error(OutputSchemaError("bad result")) == 500
    assert status_code_for_error(SchemaError("bad input")) == 400


def test_a_remote_caller_sees_the_500():
    app = create_service_app([RouteTactic()])
    remote = RemoteTactic(
        "http://svc/tactics/route",
        async_transport=httpx.ASGITransport(app=app),
    )

    with pytest.raises(RemoteTacticError) as caught:
        asyncio.run(remote.arun("a sales lead"))

    assert caught.value.status_code == 500
    assert caught.value.error_type == "OutputSchemaError"
    assert "failed validation" in caught.value.message


# 3: RemoteTactic sends headers and auth on every request, reads FastAPI's
# {"detail": "..."} error bodies, and can adopt the service's own contract.

TOKEN = "change-me"


class Ticket(BaseModel):
    text: str
    priority: int = 3


class Triage(ProtocolTactic):
    """Route a support ticket to a queue."""

    name = "triage"
    input_type = Ticket
    output_type = str

    def _run(self, ticket, *, context=None, **kwargs):
        return "oncall" if "outage" in ticket.text.lower() else "support"


def _secured_app():
    from fastapi import Request
    from fastapi.responses import JSONResponse

    app = create_tactic_app(Triage())

    @app.middleware("http")
    async def require_token(request: Request, call_next):
        if request.headers.get("authorization") != f"Bearer {TOKEN}":
            return JSONResponse({"detail": "Invalid token."}, status_code=401)
        return await call_next(request)

    return app


def _recording_transport(seen: list) -> httpx.MockTransport:
    """Answer every endpoint, recording the headers each request carried."""

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.path, dict(request.headers)))
        if request.url.path.endswith("/info"):
            return httpx.Response(200, json=Triage().info().model_dump(mode="json"))
        if request.url.path.endswith("/stream"):
            return httpx.Response(
                200,
                text='data: "oncall"\n\n',
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(
            200, json={"output": "oncall", "request_id": "r-1", "tactic": "triage"}
        )

    return httpx.MockTransport(handle)


async def _collect(items):
    return [item async for item in items]


def _call_every_endpoint(remote: RemoteTactic) -> None:
    remote.run({"text": "Outage"})
    asyncio.run(remote.arun({"text": "Outage"}))
    list(remote.stream({"text": "Outage"}))
    asyncio.run(_collect(remote.astream({"text": "Outage"})))
    asyncio.run(_collect(remote.aevents({"text": "Outage"})))
    remote.fetch_info()
    asyncio.run(remote.afetch_info())


def test_remote_tactic_headers_go_with_every_request():
    seen: list = []
    transport = _recording_transport(seen)
    remote = RemoteTactic(
        "http://triage.internal:8000",
        headers={"authorization": f"Bearer {TOKEN}"},
        transport=transport,
        async_transport=transport,
    )

    _call_every_endpoint(remote)

    assert [path for path, _headers in seen] == [
        "/run", "/run", "/stream", "/stream", "/stream", "/info", "/info"
    ]
    assert all(h["authorization"] == f"Bearer {TOKEN}" for _p, h in seen)
    assert "authorization" not in json.dumps(remote.info().model_dump(mode="json"))


def test_remote_tactic_per_call_headers_add_to_the_constructor_headers():
    seen: list = []
    transport = _recording_transport(seen)
    remote = RemoteTactic(
        "http://triage.internal:8000",
        headers={"authorization": f"Bearer {TOKEN}"},
        transport=transport,
    )

    remote.run({"text": "Outage"}, headers={"x-request-id": "r-9"})

    [(_path, headers)] = seen
    assert headers["authorization"] == f"Bearer {TOKEN}"
    assert headers["x-request-id"] == "r-9"


def test_remote_tactic_auth_goes_with_every_request():
    seen: list = []
    transport = _recording_transport(seen)
    remote = RemoteTactic(
        "http://triage.internal:8000",
        auth=("ada", "secret"),
        transport=transport,
        async_transport=transport,
    )

    _call_every_endpoint(remote)

    expected = httpx.BasicAuth("ada", "secret")._auth_header
    assert len(seen) == 7
    assert all(h["authorization"] == expected for _p, h in seen)


def test_remote_tactic_rejects_headers_that_are_not_a_mapping():
    with pytest.raises(TypeError, match="headers"):
        RemoteTactic("http://svc:8000", headers=[("authorization", "x")])


def test_remote_tactic_reaches_a_secured_service():
    app = _secured_app()
    remote = RemoteTactic(
        "http://triage.internal:8000",
        headers={"authorization": f"Bearer {TOKEN}"},
        async_transport=httpx.ASGITransport(app=app),
    )

    assert asyncio.run(remote.arun({"text": "Outage"})) == "oncall"
    assert asyncio.run(remote.afetch_info()).name == "triage"
    # Past the middleware (not a 401): Triage does not stream.
    with pytest.raises(RemoteTacticError) as caught:
        asyncio.run(_collect(remote.aevents({"text": "Outage"})))
    assert caught.value.status_code == 400
    assert caught.value.error_type == "TacticUnsupportedError"


def test_a_fastapi_detail_string_becomes_the_error_message():
    app = _secured_app()
    remote = RemoteTactic(
        "http://triage.internal:8000",
        async_transport=httpx.ASGITransport(app=app),
    )

    for call in (
        lambda: remote.arun({"text": "Outage"}),
        lambda: remote.afetch_info(),
        lambda: _collect(remote.aevents({"text": "Outage"})),
    ):
        with pytest.raises(RemoteTacticError) as caught:
            asyncio.run(call())
        assert caught.value.status_code == 401
        assert caught.value.message == "Invalid token."
        assert caught.value.error_type is None
        assert str(caught.value) == "Remote tactic returned HTTP 401: Invalid token."
    error = RemoteTacticError(403, detail={"detail": "Forbidden."})
    assert error.message == "Forbidden."


def test_from_service_adopts_the_service_contract():
    from lllm.runtimes.native import tactic_as_function
    from lllm.runtimes.pydantic_ai import tactic_as_tool

    seen: list = []
    remote = RemoteTactic.from_service(
        "http://triage.internal:8000",
        headers={"authorization": f"Bearer {TOKEN}"},
        transport=_recording_transport(seen),
    )
    expected = Triage().info()

    info = remote.info()
    assert [path for path, _headers in seen] == ["/info"]
    assert seen[0][1]["authorization"] == f"Bearer {TOKEN}"
    assert remote.tactic_name == "triage"
    assert info.description == "Route a support ticket to a queue."
    assert info.input_schema == expected.input_schema
    assert info.output_schema == expected.output_schema
    assert info.runtime_kind == "http"
    assert info.service_ref == "http://triage.internal:8000"
    assert remote.capabilities() == {"run", "arun"}
    assert not remote.supports("stream")

    tool = tactic_as_function(remote, parameter_mode="kwargs")
    assert tool.name == "triage"
    assert tool.description == "Route a support ticket to a queue."
    assert sorted(tool.properties) == ["priority", "text"]
    assert tool.required == ["text"]
    task_tool = tactic_as_function(remote)
    assert task_tool.properties["task"]["required"] == ["text"]
    assert tactic_as_tool(remote).__doc__ == "Route a support ticket to a queue."

    named = RemoteTactic.from_service(
        "http://triage.internal:8000",
        name="router",
        transport=_recording_transport(seen),
    )
    assert named.tactic_name == "router"
    assert named.info().description == "Route a support ticket to a queue."


def test_afrom_service_adopts_the_contract_of_a_running_service():
    app = _secured_app()

    async def main():
        remote = await RemoteTactic.afrom_service(
            "http://triage.internal:8000",
            headers={"authorization": f"Bearer {TOKEN}"},
            async_transport=httpx.ASGITransport(app=app),
        )
        return remote, await remote.arun({"text": "Total outage"})

    remote, output = asyncio.run(main())

    assert output == "oncall"
    assert remote.tactic_name == "triage"
    assert remote.info().input_schema["required"] == ["text"]


class StreamOnly(ProtocolTactic):
    """Stream the steps of a triage; the last step fails."""

    name = "steps"

    def _run(self, text, **kwargs):
        return "unused"

    def stream(self, text, **kwargs):
        yield "reading"
        raise RuntimeError("upstream broke")


class EventsOnly(ProtocolTactic):
    name = "events"

    def _run(self, text, **kwargs):
        return "unused"

    async def aevents(self, text, **kwargs):
        from lllm import TacticEvent

        yield TacticEvent(kind="step", data="reading")


@pytest.mark.parametrize("upstream", [StreamOnly, EventsOnly])
def test_an_adopted_streaming_contract_keeps_stream_and_events(upstream):
    tactic = upstream()
    assert tactic.capabilities() & {"stream", "events"} != {"stream", "events"}

    app = create_tactic_app(tactic)
    remote = asyncio.run(
        RemoteTactic.afrom_service(
            "http://svc:8000", async_transport=httpx.ASGITransport(app=app)
        )
    )

    assert remote.supports("stream") and remote.supports("events")
    assert remote.info().capabilities == ("arun", "events", "run", "stream")


def test_a_proxied_adopted_streaming_tactic_keeps_an_error_events_kind():
    from lllm import proxy_tactic

    remote = asyncio.run(
        RemoteTactic.afrom_service(
            "http://svc:8000",
            async_transport=httpx.ASGITransport(app=create_tactic_app(StreamOnly())),
        )
    )
    proxied = proxy_tactic(remote)

    events = asyncio.run(_collect(proxied.aevents("Outage")))

    assert [event.kind for event in events] == ["message", "error"]
    assert events[-1].data == {"message": "upstream broke"}
    served = TestClient(create_tactic_app(proxied))
    with served.stream("POST", "/stream", json={"input": "Outage"}) as response:
        kinds = [
            json.loads(line.removeprefix("data: "))["kind"]
            for line in response.iter_lines()
            if line.startswith("data: ")
        ]
    assert kinds == ["message", "error"]


def test_a_contract_supplies_examples_when_the_tactic_has_none():
    example = {"input": {"text": "Outage"}, "output": "oncall"}
    contract = Triage().info().model_copy(update={"examples": [example]})

    remote = RemoteTactic("http://triage.internal:8000", contract=contract)

    assert remote.info().examples == [example]
    remote.info().examples[0]["output"] = "changed"
    assert remote.info().examples == [example]
    assert contract.examples == [example]


def test_a_given_contract_is_adopted_where_the_tactic_has_nothing_of_its_own():
    contract = Triage().info()
    remote = RemoteTactic(
        "http://triage.internal:8000",
        output_type=int,
        contract=contract.model_dump(mode="json"),
    )

    info = remote.info()
    assert remote.tactic_name == "triage"
    assert info.description == contract.description
    assert info.input_schema == contract.input_schema
    assert info.output_schema == {"type": "integer"}
    with pytest.raises(TypeError, match="contract"):
        RemoteTactic("http://svc:8000", contract="triage")


# 4: tactic_as_tool(..., asynchronous=True) and Tactic.as_tool(...) make an
# async tool that awaits arun(), so a wrapped Pydantic AI agent can be a
# tool of another Pydantic AI agent.


def _planner_model(tool_name: str, args: dict):
    from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart

    def model(messages, info):
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart(tool_name, args)])
        return ModelResponse(parts=[TextPart(f"got: {messages[-1].parts[0].content}")])

    return model


def _changelog_tactic():
    from pydantic_ai import Agent
    from pydantic_ai.models.test import TestModel

    from lllm import PydanticAITactic

    agent = Agent(TestModel(custom_output_text="Fixed: uploads retry."), name="changelog")
    return PydanticAITactic(agent, input_type=str)


def test_an_async_tool_lets_one_pydantic_ai_agent_call_another():
    import inspect

    from pydantic_ai import Agent
    from pydantic_ai.exceptions import UserError
    from pydantic_ai.models.function import FunctionModel

    from lllm import tactic_as_tool

    changelog = _changelog_tactic()
    model = FunctionModel(_planner_model("changelog", {"task": "Retry uploads"}))

    tool = tactic_as_tool(changelog, asynchronous=True)
    assert inspect.iscoroutinefunction(tool)
    assert tool.__name__ == "changelog"
    assert str(inspect.signature(tool)) == "(task: str) -> str"
    planner = Agent(model, tools=[tool])
    assert planner.run_sync("Write the entry").output == "got: Fixed: uploads retry."

    # The synchronous tool calls run_sync() inside the planner's run.
    sync_planner = Agent(model, tools=[tactic_as_tool(changelog)])
    with pytest.raises(UserError):
        sync_planner.run_sync("Write the entry")


def test_async_tools_await_arun_in_both_parameter_modes():
    import inspect

    from lllm import tactic_as_tool

    calls = []

    class Shout(ProtocolTactic):
        name = "shout"
        input_type = Ticket
        output_type = str

        def _run(self, ticket, **kwargs):
            raise AssertionError("an async tool must not call run()")

        async def _arun(self, ticket, **kwargs):
            calls.append(ticket)
            return ticket.text.upper()

    task_tool = tactic_as_tool(Shout(), asynchronous=True)
    kwargs_tool = tactic_as_tool(Shout(), asynchronous=True, parameter_mode="kwargs")

    assert inspect.iscoroutinefunction(kwargs_tool)
    assert "(*, text: str, priority: int = 3) -> str" == str(inspect.signature(kwargs_tool))
    assert asyncio.run(task_tool({"text": "hi"})) == "HI"
    assert asyncio.run(kwargs_tool(text="yo")) == "YO"
    assert calls == [Ticket(text="hi"), Ticket(text="yo")]
    with pytest.raises(TypeError, match="asynchronous"):
        tactic_as_tool(Shout(), asynchronous="yes")


def test_native_as_tool_makes_an_async_tool_too():
    import inspect

    class Upper(NativeTactic):
        """Upper-case a text."""

        name = "upper"
        input_type = str
        output_type = str

        def call(self, text: str) -> str:
            return text.upper()

    tool = Upper().as_tool(asynchronous=True)

    assert inspect.iscoroutinefunction(tool)
    assert tool.__doc__ == "Upper-case a text."
    assert asyncio.run(tool("hi")) == "HI"
    assert not inspect.iscoroutinefunction(Upper().as_tool())


# 5: the config template's URLs use the ports `lllm serve` listens on.

PORTS_MANIFEST = """
[package]
org = "demo"
name = "ports"
version = "0.1.0"

[tactics.alpha]
entry = "fix3_ports.tactics:Echo"
metadata = {port = 8100}

[tactics.beta]
entry = "fix3_ports.tactics:Echo"

[tactics.gamma]
entry = "fix3_ports.tactics:Echo"

[tactics.delta]
entry = "fix3_ports.tactics:Echo"

[tactics.epsilon]
entry = "fix3_ports.tactics:Echo"
port = 8200

[tactics.zeta]
entry = "fix3_ports.tactics:Echo"

[tactics.eta]
entry = "fix3_ports.tactics:Echo"

[services.api]
tactic = "alpha"
metadata = {port = 9000}

[services.plain]
tactic = "alpha"

[services.multi]
entry = "fix3_ports.app:create_app"
tactics = ["gamma", "beta"]

[services.bundle]
entry = "fix3_ports.app:create_app"
tactics = ["beta"]
metadata = {port = 9300}

[services.second]
tactic = "delta"
metadata = {port = 9500}

[services.third]
entry = "fix3_ports.app:create_app"
tactics = ["delta"]
metadata = {port = 9700}

[services.direct]
entry = "fix3_ports.app:create_app"
port = 9600

[services.zz_first]
entry = "fix3_ports.app:create_app"
tactics = ["zeta"]
metadata = {port = 9990}

[services.aa_second]
tactic = "zeta"
metadata = {port = 9100}

[services.eta_host]
tactic = "eta"
port = 9800
"""


def _ports_package(tmp_path: Path) -> Path:
    package = tmp_path / "ports"
    _write(package / "lllm.toml", PORTS_MANIFEST)
    _write(package / "fix3_ports" / "__init__.py", "")
    _write(
        package / "fix3_ports" / "tactics.py",
        """
class Echo:
    def run(self, value, *, context=None):
        return value
""",
    )
    _write(
        package / "fix3_ports" / "app.py",
        """
from fastapi import FastAPI


def create_app():
    return FastAPI()
""",
    )
    return package


def _template_ports(package: Path) -> dict[str, int]:
    from urllib.parse import urlsplit

    try:
        import tomllib
    except ImportError:  # Python 3.10
        import tomli as tomllib

    from lllm.hub import load_manifest, record_from_manifest, render_config_template

    record = record_from_manifest(load_manifest(package))
    config = tomllib.loads(render_config_template(record))
    ports = {}
    for ref, binding in config["refs"].items():
        section, name = ref.rsplit("/", 2)[-2:]
        ports[f"{section}.{name}"] = urlsplit(binding["url"]).port
    for name, settings in config["services"].items():
        assert settings["port"] == ports[f"services.{name}"]
    return ports


def test_config_template_ports_match_the_ports_lllm_serve_uses(tmp_path):
    from lllm.launch import load_launch_app

    package = _ports_package(tmp_path)

    ports = _template_ports(package)
    served = {
        resource: load_launch_app(package, resource=resource).port
        for resource in ports
    }

    assert ports == served
    assert ports == {
        "tactics.alpha": 8100,  # its own port, though services.api serves it
        "tactics.beta": 9300,  # the first serving service with a port
        "tactics.gamma": 8000,  # served only by a service without one
        "tactics.delta": 9500,  # two serving services have ports: the first
        "tactics.epsilon": 8200,  # a port written in the table itself
        # The first serving service in manifest order wins, though it lists
        # the tactic in tactics, has the higher port, and sorts last.
        "tactics.zeta": 9990,
        "tactics.eta": 9800,  # its only service writes its port in its table
        "services.api": 9000,
        "services.plain": 8000,
        "services.multi": 8000,
        "services.bundle": 9300,
        "services.second": 9500,
        "services.third": 9700,
        "services.direct": 9600,  # a port written in the table itself
        "services.zz_first": 9990,
        "services.aa_second": 9100,
        "services.eta_host": 9800,
    }


# 6: install_package accepts the underscored names `lllm create` produces.


@pytest.fixture
def glossary_zip(tmp_path, monkeypatch):
    """A zip of a package made by `lllm create`, and a project to install into."""

    import shutil

    from lllm import create_project
    from lllm.hub import init_package

    monkeypatch.delenv("LLLM_CONFIG", raising=False)
    result = create_project("minimal", "my-glossary", directory=tmp_path / "made")
    assert result.package_name == "my_glossary"
    archive = shutil.make_archive(str(tmp_path / "glossary"), "zip", result.path)
    project = tmp_path / "project"
    init_package(project, name="project")
    monkeypatch.chdir(project)
    return Path(archive)


@pytest.mark.parametrize("alias", ["team_glossary", "glossary_v2", "g"])
def test_install_package_accepts_an_underscored_alias(glossary_zip, alias):
    from lllm import Runtime, install_package

    runtime = Runtime()
    path = install_package(glossary_zip, alias=alias, scope="project", runtime=runtime)

    assert path == Path.cwd() / "lllm_packages" / alias
    assert f'name = "{alias}"' in (path / "lllm.toml").read_text(encoding="utf-8")
    assert f"{alias}.prompts:assistant/system" in runtime.keys()


def test_install_package_keeps_the_created_name_without_an_alias(glossary_zip):
    from lllm import install_package

    path = install_package(glossary_zip, scope="project", load=False)

    assert path.name == "my_glossary"


@pytest.mark.parametrize(
    "alias", ["Team", "team glossary", "team.glossary", "_team", "team_", "-x", "a/b"]
)
def test_install_package_still_rejects_invalid_aliases(glossary_zip, alias):
    from lllm import install_package

    with pytest.raises(ValueError, match="not a valid package name"):
        install_package(glossary_zip, alias=alias, scope="project", load=False)


# 7: every heading of the agent card follows a blank line, tags or not.


@pytest.mark.parametrize(
    "card",
    [
        'tags = ["support", "routing"]',
        'safety = "Reads tickets only."\nlatency = "About a second."',
        'tags = ["support"]\nsuggested_commands = ["lllm serve ./triage"]',
        'summary = "Routes tickets."',
    ],
)
@pytest.mark.parametrize("api_keys", [False, True])
def test_agent_card_headings_follow_a_blank_line(tmp_path, card, api_keys):
    from lllm.hub import load_manifest, record_from_manifest, render_agent_card

    requirements = (
        '\n[requirements.api_keys]\nOPENAI_API_KEY = "Model access."\n'
        if api_keys
        else ""
    )
    package = tmp_path / "triage"
    _write(
        package / "lllm.toml",
        f"""
[package]
org = "acme"
name = "triage"

[card]
{card}

[tactics.triage]
entry = "main:create_tactic"
{requirements}""",
    )

    lines = render_agent_card(record_from_manifest(load_manifest(package))).splitlines()

    headings = [index for index, line in enumerate(lines) if line.startswith("## ")]
    assert headings
    for index in headings:
        assert lines[index - 1] == "", lines[index - 1 : index + 1]
    assert "## Resources" in lines


# 8: serving an undeclared channel is an error, and inspecting a channel
# package creates no store.

CHANNELS_MANIFEST = """
[package]
org = "demo"
name = "events"
kind = "channel"
primary = "channels.events"

[channels.events]
form = "log"
description = "Input events."

[channels.alerts]
form = "log"
"""


@pytest.fixture
def channel_package(tmp_path):
    package = tmp_path / "events"
    _write(package / "lllm.toml", CHANNELS_MANIFEST)
    return package


@pytest.fixture
def fake_sssn(monkeypatch):
    """Stand in for sssn: opening a package store creates its folder."""

    import types

    from fastapi import FastAPI

    opened = []

    def open_package_store(manifest, *, store=None):
        root = Path(store) if store is not None else manifest.base_dir / ".lllm" / "network"
        root.mkdir(parents=True, exist_ok=True)  # as sssn's LocalStore does
        opened.append(root)
        return {"root": root, "channels": sorted(manifest.channels)}

    def create_app(store):
        app = FastAPI()

        @app.get("/channels")
        def channels():
            return store["channels"]

        return app

    package = types.ModuleType("sssn")
    package.__path__ = []
    resources = types.ModuleType("sssn.resources")
    resources.open_package_store = open_package_store
    server = types.ModuleType("sssn.server")
    server.create_app = create_app
    for name, module in {
        "sssn": package,
        "sssn.resources": resources,
        "sssn.server": server,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    return opened


def test_launching_an_undeclared_channel_is_an_error(channel_package, uninstalled):
    from lllm.launch import LaunchError, load_launch_app

    uninstalled("sssn")  # the name is checked before sssn is needed

    with pytest.raises(LaunchError) as caught:
        load_launch_app(channel_package, resource="channels.nope")

    assert str(caught.value) == (
        "channel not found in package: nope; declared channels: events, alerts"
    )


def test_serve_rejects_an_undeclared_channel(channel_package, fake_sssn, served, capsys):
    with pytest.raises(SystemExit) as caught:
        main(["serve", str(channel_package), "--resource", "channels.nope"])

    assert caught.value.code == 2
    assert "channel not found in package: nope" in capsys.readouterr().err
    assert served == [] and fake_sssn == []
    assert not (channel_package / ".lllm").exists()


def test_serve_still_launches_a_declared_channel(channel_package, fake_sssn, served):
    assert main(["serve", str(channel_package), "--resource", "channels.alerts"]) == 0

    [(app, _options)] = served
    assert fake_sssn == [channel_package / ".lllm" / "network"]
    assert TestClient(app).get("/channels").json() == ["alerts", "events"]


@pytest.mark.parametrize("store", [None, "elsewhere"])
def test_inspecting_a_channel_package_creates_nothing(
    channel_package, fake_sssn, tmp_path, capsys, store
):
    args = ["inspect", str(channel_package)]
    if store is not None:
        args += ["--store", str(tmp_path / store)]

    assert main(args) == 0

    out = capsys.readouterr().out
    assert "demo/events:channels.events (channel)" in out
    assert "GET /channels" in out
    assert not (channel_package / ".lllm").exists()
    assert not (tmp_path / "elsewhere").exists()
    assert fake_sssn and not any(root.exists() for root in fake_sssn)


def test_inspect_help_hides_store_but_inspect_still_accepts_it(
    channel_package, fake_sssn, tmp_path, monkeypatch, capsys
):
    for command, listed in (("inspect", False), ("serve", True)):
        assert ("--store" in _help(command, monkeypatch, capsys)) is listed

    args = ["inspect", str(channel_package), "--store", str(tmp_path / "s")]
    assert main(args) == 0
    assert not (tmp_path / "s").exists()


# Round 4, 1: a tactic with routes and router attributes is a tactic, not
# an app, as in 0.4.0.

DISPATCH_MODULE = """
from lllm import Tactic


class Dispatch(Tactic):
    name = "dispatch"
    description = "Route a ticket by keyword."
    routes = {"refund": "billing", "crash": "bugs"}
    router = "keyword"

    def call(self, text: str) -> str:
        for word, queue in self.routes.items():
            if word in text:
                return queue
        return "general"


dispatch = Dispatch()
"""


def test_a_routing_tactic_is_not_an_asgi_app():
    from fastapi import FastAPI
    from starlette.applications import Starlette

    from lllm.hub.entrypoints import is_asgi_app

    namespace: dict = {}
    exec(DISPATCH_MODULE, namespace)

    assert not is_asgi_app(namespace["Dispatch"])
    assert not is_asgi_app(namespace["dispatch"])
    assert not is_asgi_app(FastAPI)  # a class, not an app
    assert is_asgi_app(FastAPI()) and is_asgi_app(Starlette())


def test_without_starlette_an_app_is_a_callable_with_routes_and_router(uninstalled):
    from lllm.hub.entrypoints import is_asgi_app

    namespace: dict = {}
    exec(DISPATCH_MODULE, namespace)
    uninstalled("starlette")

    class App:
        routes: list = []
        router = object()

        def __call__(self, scope, receive, send):
            return None

    assert is_asgi_app(App())
    assert not is_asgi_app(App)
    assert not is_asgi_app(namespace["Dispatch"])
    assert not is_asgi_app(namespace["dispatch"])


def test_validate_and_package_serve_accept_a_routing_tactic(tmp_path):
    from lllm.hub import LocalHub, validate_package
    from lllm.launch import load_launch_app

    package = tmp_path / "dispatch"
    _write(
        package / "lllm.toml",
        """
[package]
org = "demo"
name = "dispatch"

[tactics.by_class]
entry = "fix3_dispatch:Dispatch"

[tactics.by_instance]
entry = "fix3_dispatch:dispatch"
""",
    )
    _write(package / "fix3_dispatch.py", DISPATCH_MODULE)

    report = validate_package(package)

    assert report.ok, report.issues
    assert not any(issue.code == "tactic_entry_is_app" for issue in report.issues)
    LocalHub(tmp_path / "hub").publish(package)
    for resource in ("tactics.by_class", "tactics.by_instance"):
        app = load_launch_app(package, resource=resource).app
        response = TestClient(app).post("/run", json={"input": "refund please"})
        assert response.json()["output"] == "billing"


@pytest.mark.parametrize("attr", ["Dispatch", "dispatch"])
def test_the_cli_serves_and_inspects_a_routing_tactic(
    tmp_path, monkeypatch, served, capsys, attr
):
    _write(tmp_path / "fix3_dispatch.py", DISPATCH_MODULE)
    monkeypatch.syspath_prepend(str(tmp_path))

    assert load_tactic_entrypoint(f"fix3_dispatch:{attr}").tactic_name == "dispatch"
    assert main(["inspect", f"fix3_dispatch:{attr}"]) == 0
    assert capsys.readouterr().out.splitlines()[0] == "dispatch (native)"
    assert main(["serve", f"fix3_dispatch:{attr}"]) == 0

    [(app, _options)] = served
    response = TestClient(app).post("/run", json={"input": "the app crashed"})
    assert response.json()["output"] == "bugs"


# Review 6, final rule: a web page cannot make a visitor's browser run a
# served tactic. A cross-site POST that is not JSON, or a cross-site GET to
# a custom endpoint, is refused; PUT, PATCH, and DELETE are left to CORS.

EVIL = "https://evil.example"


class Echo(ProtocolTactic):
    name = "echo"

    def _run(self, value, *, context=None, **kwargs):
        return value

    def stream(self, value, **kwargs):
        yield value

    @endpoint.post("/echo/shout")
    def shout(self, value):
        return str(value).upper()

    @endpoint.get("/echo/status")
    def status(self, value):
        return {"query": value}

    @endpoint.put("/echo/item")
    def put_item(self, value):
        return {"put": value}

    @endpoint.patch("/echo/item")
    def patch_item(self, value):
        return {"patched": value}

    @endpoint.delete("/echo/item")
    def delete_item(self, value):
        return {"deleted": value}


def _local_client(app=None) -> TestClient:
    return TestClient(app or create_tactic_app(Echo()), base_url="http://127.0.0.1:8000")


def _assert_refused(response, endpoint_name: str) -> None:
    assert response.status_code == 403
    error = response.json()["detail"]["error"]
    assert error["type"] == "CrossSiteRequestError"
    assert error["tactic"] == "echo"
    assert error["endpoint"] == endpoint_name
    assert error["request_id"]


# Bodies a browser posts to another site without a preflight, and the
# content type each one must really be sent with.
SIMPLE_BODIES = {
    "text": ({"content": "hi", "headers": {"content-type": "text/plain"}}, "text/plain"),
    "form": ({"data": {"input": "hi"}}, "application/x-www-form-urlencoded"),
    "multipart": ({"files": {"input": ("hi.txt", b"hi")}}, "multipart/form-data"),
    "no-content-type": ({"content": b"hi"}, None),
}

# Ways a request is cross-site. Sec-Fetch-Site decides alone when present.
CROSS_SITE = {
    "sec-cross-site": {"sec-fetch-site": "cross-site", "origin": EVIL},
    "sec-same-site": {"sec-fetch-site": "same-site", "origin": "http://127.0.0.1:8000"},
    "sec-only": {"sec-fetch-site": "cross-site"},  # no Origin, as on a no-cors GET
    "other-site": {"origin": EVIL},
    "other-port": {"origin": "http://127.0.0.1:3000"},
    "null": {"origin": "null"},  # a sandboxed page or a file
    "not-web": {"origin": "chrome-extension://abcdef"},
}


def _post_simple(client, path, body, signal):
    kwargs, content_type = SIMPLE_BODIES[body]
    headers = {**kwargs.get("headers", {}), **CROSS_SITE[signal]}
    sent = {key: value for key, value in kwargs.items() if key != "headers"}
    response = client.post(path, headers=headers, **sent)
    # The case really sent the body it is named after.
    actual = response.request.headers.get("content-type")
    if content_type is None:
        assert actual is None
    else:
        assert actual.split(";")[0] == content_type
    return response


@pytest.mark.parametrize("body", list(SIMPLE_BODIES))
@pytest.mark.parametrize("signal", list(CROSS_SITE))
def test_a_cross_site_post_that_is_not_json_is_refused(body, signal):
    response = _post_simple(_local_client(), "/run", body, signal)

    _assert_refused(response, "run")
    assert "JSON" in response.json()["detail"]["error"]["message"]


@pytest.mark.parametrize(
    "path",
    ["/run", "/stream", "/tactics/echo/run", "/tactics/echo/stream", "/echo/shout"],
)
def test_every_tactic_post_route_refuses_cross_site_text(path):
    response = _post_simple(_local_client(), path, "text", "other-site")

    _assert_refused(response, path.rsplit("/", 1)[-1])


@pytest.mark.parametrize(
    "headers",
    [{"sec-fetch-site": "cross-site"}, {"sec-fetch-site": "same-site"}, {"origin": EVIL}],
    ids=["sec-cross-site", "sec-same-site", "origin"],
)
def test_a_cross_site_get_to_a_custom_endpoint_is_refused(headers):
    response = _local_client().get("/echo/status?q=1", headers=headers)

    _assert_refused(response, "status")
    assert "custom endpoint" in response.json()["detail"]["error"]["message"]


@pytest.mark.parametrize(
    "headers",
    [{}, {"sec-fetch-site": "same-origin"}, {"sec-fetch-site": "none"}],
    ids=["not-a-browser", "same-origin", "typed-url"],
)
def test_a_get_to_a_custom_endpoint_from_this_origin_runs(headers):
    response = _local_client().get("/echo/status?q=1", headers=headers)

    assert response.json() == {"query": {"q": "1"}}


def test_cors_approved_cross_origin_put_patch_and_delete_are_allowed():
    from starlette.middleware.cors import CORSMiddleware

    app = create_tactic_app(Echo())
    app.add_middleware(
        CORSMiddleware, allow_origins=[EVIL], allow_methods=["*"], allow_headers=["*"]
    )
    client = _local_client(app)
    preflight = client.options(
        "/echo/item",
        headers={"origin": EVIL, "access-control-request-method": "DELETE"},
    )
    assert preflight.status_code == 200
    assert preflight.headers["access-control-allow-origin"] == EVIL

    headers = {"origin": EVIL, "sec-fetch-site": "cross-site", "content-type": "text/plain"}
    put = client.put("/echo/item", content="hi", headers=headers)
    patch = client.patch("/echo/item", content="hi", headers=headers)
    delete = client.request("DELETE", "/echo/item", content="hi", headers=headers)

    assert put.json() == {"put": "hi"}
    assert patch.json() == {"patched": "hi"}
    assert delete.json() == {"deleted": "hi"}


def test_a_same_origin_request_behind_a_host_rewriting_proxy_is_allowed():
    client = TestClient(create_tactic_app(Echo()))
    headers = {
        "content-type": "text/plain",
        "origin": "https://api.example.com",
        "host": "app.internal:8000",  # rewritten by the proxy
    }

    by_origin = client.post("/run", content="hi", headers=headers)
    by_fetch_site = client.post(
        "/run", content="hi", headers={**headers, "sec-fetch-site": "same-origin"}
    )

    assert by_origin.status_code == 403  # without the header, Origin decides
    assert by_fetch_site.json()["output"] == "hi"


@pytest.mark.parametrize(
    "headers",
    [
        {"content-type": "application/json", "origin": EVIL, "sec-fetch-site": "cross-site"},
        {"content-type": "application/vnd.api+json", "origin": EVIL},
        {"content-type": "application/json; charset=utf-8", "sec-fetch-site": "cross-site"},
        {"content-type": "text/plain"},  # not a browser
        {"content-type": "text/plain", "origin": "http://127.0.0.1:8000"},
        {"content-type": "text/plain", "sec-fetch-site": "same-origin", "origin": EVIL},
        {"content-type": "text/plain", "sec-fetch-site": "none"},
    ],
    ids=[
        "json",
        "plus-json",
        "json-charset",
        "not-a-browser",
        "same-origin",
        "sec-fetch-site-decides",
        "sec-none",
    ],
)
def test_json_non_browser_and_same_origin_posts_still_run(headers):
    content = '"hi"' if "json" in headers["content-type"] else "hi"

    response = _local_client().post("/run", content=content, headers=headers)

    assert response.status_code == 200
    assert response.json()["output"] == "hi"


@pytest.mark.parametrize(
    ("origin", "host"),
    [
        ("http://localhost", "localhost"),  # default ports on both sides
        ("http://localhost", "localhost:80"),
        ("https://api.example.com", "api.example.com"),  # behind a TLS proxy
        ("https://API.example.com:443", "api.example.com"),
        ("http://[::1]:8000", "[::1]:8000"),
    ],
)
def test_the_same_origin_spelled_differently_is_not_cross_site(origin, host):
    client = TestClient(create_tactic_app(Echo()))

    response = client.post(
        "/run",
        content="hi",
        headers={"content-type": "text/plain", "origin": origin, "host": host},
    )

    assert response.status_code == 200


@pytest.mark.parametrize(
    "path", ["/info", "/health", "/docs", "/openapi.json", "/tactics", "/tactics/echo/info"]
)
def test_read_routes_answer_cross_site_requests(path):
    headers = {"origin": EVIL, "sec-fetch-site": "cross-site"}

    assert _local_client().get(path, headers=headers).status_code == 200


OPT_OUT_APPS = {
    "create_tactic_app": lambda: create_tactic_app(
        Echo(), allow_cross_site_requests=True
    ),
    "create_service_app": lambda: create_service_app(
        [Echo()], allow_cross_site_requests=True
    ),
}

# Every handler that checks cross-site requests, by app: each checks on its
# own, so each must honor the opt-out.
GUARDED_ROUTES = [
    ("create_tactic_app", "/run"),
    ("create_tactic_app", "/stream"),
    ("create_tactic_app", "/tactics/echo/run"),
    ("create_tactic_app", "/tactics/echo/stream"),
    ("create_tactic_app", "/echo/shout"),
    ("create_tactic_app", "/echo/status"),
    ("create_service_app", "/tactics/echo/run"),
    ("create_service_app", "/tactics/echo/stream"),
    ("create_service_app", "/echo/shout"),
    ("create_service_app", "/echo/status"),
]


@pytest.mark.parametrize(("app_factory", "path"), GUARDED_ROUTES)
def test_allow_cross_site_requests_turns_off_every_check(app_factory, path):
    client = _local_client(OPT_OUT_APPS[app_factory]())
    signal = {"origin": EVIL, "sec-fetch-site": "cross-site"}

    if path == "/echo/status":  # the custom GET endpoint
        response = client.get(f"{path}?q=1", headers=signal)
        assert response.status_code == 200
        assert response.json() == {"query": {"q": "1"}}
        return
    response = client.post(
        path, content="hi", headers={**signal, "content-type": "text/plain"}
    )

    assert response.status_code == 200
    if path.endswith("/stream"):
        assert '"data": "hi"' in response.text
    elif path == "/echo/shout":
        assert response.json() == "HI"
    else:
        assert response.json()["output"] == "hi"


def test_allow_cross_site_requests_must_be_a_boolean():
    with pytest.raises(TypeError, match="allow_cross_site_requests"):
        create_tactic_app(Echo(), allow_cross_site_requests="yes")
    with pytest.raises(TypeError, match="allow_cross_site_requests"):
        create_service_app([Echo()], allow_cross_site_requests=1)


# Review 7: a body is decoded strictly as UTF-8 before it is parsed as
# JSON. Other bytes are a 400, except an application/octet-stream body,
# which the tactic gets as bytes, as in 0.4.0. A result that cannot be
# encoded is a 500 in the services envelope, never a bare traceback.

BODY_ROUTES = ["/run", "/stream", "/tactics/echo/run", "/echo/shout"]


def _assert_bad_body(response) -> None:
    assert response.status_code == 400
    error = response.json()["detail"]["error"]
    assert error["type"] == "SchemaError"
    assert error["message"] == (
        "Request body must be JSON or UTF-8 text; send binary data as "
        "application/octet-stream."
    )
    assert error["tactic"] == "echo"


@pytest.mark.parametrize("path", BODY_ROUTES)
@pytest.mark.parametrize("content_type", ["text/plain", "application/json", None])
def test_a_body_that_is_neither_json_nor_utf8_text_is_a_400(path, content_type):
    client = TestClient(create_tactic_app(Echo()), raise_server_exceptions=False)
    headers = {"content-type": content_type} if content_type else {}

    response = client.post(path, content=b"\xff\xfe binary", headers=headers)

    _assert_bad_body(response)


@pytest.mark.parametrize("path", BODY_ROUTES)
def test_json_with_utf8_encoded_surrogates_is_a_400(path):
    client = TestClient(create_tactic_app(Echo()), raise_server_exceptions=False)
    body = b'{"input": "\xed\xa0\x80"}'  # json.loads(bytes) would accept it

    response = client.post(path, content=body, headers={"content-type": "application/json"})

    _assert_bad_body(response)


@pytest.mark.parametrize("path", BODY_ROUTES)
@pytest.mark.parametrize(
    "body",
    [
        rb'{"input": "\ud800"}',  # plain ASCII, so it passes the UTF-8 decode
        rb'{"input": {"\udc00": 1}}',  # in a key
        rb'{"input": "x", "context": {"metadata": {"note": "a\udfffb"}}}',
        rb'{"input": ["a", "\ud800"]}',  # in a list
        rb'"\ud800"',  # a bare JSON string, not an envelope
    ],
    ids=["value", "key", "context", "list", "bare-string"],
)
def test_json_with_an_escaped_lone_surrogate_is_a_400(path, body):
    client = TestClient(create_tactic_app(Echo()), raise_server_exceptions=False)

    response = client.post(path, content=body, headers={"content-type": "application/json"})

    assert response.status_code == 400
    error = response.json()["detail"]["error"]
    assert error["type"] == "SchemaError"
    assert "lone surrogate" in error["message"]


def test_an_escaped_surrogate_pair_is_fine():
    client = TestClient(create_tactic_app(Echo()))

    response = client.post(
        "/run",
        content=rb'{"input": "\ud83d\ude00"}',  # one escaped pair
        headers={"content-type": "application/json"},
    )

    assert response.json()["output"] == "\U0001f600"


def test_an_error_message_with_a_lone_surrogate_keeps_the_json_envelope():
    class Fails(ProtocolTactic):
        name = "fails"

        def _run(self, value, **kwargs):
            raise ValueError("bad \ud800 value")

    client = TestClient(create_tactic_app(Fails()), raise_server_exceptions=False)

    response = client.post("/run", json={"input": "x", "context": {"request_id": "r-9"}})

    assert response.status_code == 500
    assert response.headers["content-type"] == "application/json"
    error = response.json()["detail"]["error"]
    assert error["type"] == "ValueError"
    assert error["message"] == "bad \\ud800 value"
    assert error["request_id"] == "r-9"


# Round 5, 4: an unknown tactic name the error envelope cannot carry as
# its tactic still gets the 404 envelope, as in AAAX.


@pytest.mark.parametrize("name", ["a;b", "a:b", "a%b", "a\\b", " "])
@pytest.mark.parametrize(
    ("method", "route"), [("POST", "run"), ("POST", "stream"), ("GET", "info")]
)
def test_an_unknown_tactic_name_of_any_shape_is_a_404_envelope(name, method, route):
    from urllib.parse import quote

    client = TestClient(create_tactic_app(Echo()), raise_server_exceptions=False)

    response = client.request(method, f"/tactics/{quote(name, safe='')}/{route}", json={"input": "x"})

    assert response.status_code == 404
    error = response.json()["detail"]["error"]
    assert error["type"] == "TacticNotFound"
    assert error["message"] == f"Tactic not found: {name}"
    assert error["tactic"] is None


def test_an_unknown_tactic_name_the_envelope_can_carry_keeps_it():
    client = TestClient(create_tactic_app(Echo()))

    error = client.post("/tactics/nope/run", json={"input": "x"}).json()["detail"]["error"]

    assert (error["type"], error["tactic"]) == ("TacticNotFound", "nope")


# Round 5, 1: a body whose first value is an array or object nested more
# than 500 levels deep (brackets in strings aside) is a 400, as in AAAX,
# before it is parsed. A deep value could otherwise crash the server when
# code walks it recursively on a small thread stack.

TOO_DEEP = "Request body is nested more than 500 levels deep."


class Ok(ProtocolTactic):
    name = "ok"

    def _run(self, value, **kwargs):
        return "ok"


def _nested(depth: int) -> bytes:
    return ("[" * depth + "]" * depth).encode()


@pytest.mark.parametrize("path", BODY_ROUTES)
def test_a_body_nested_501_levels_is_a_400(path):
    client = TestClient(create_tactic_app(Echo()), raise_server_exceptions=False)

    response = client.post(path, content=_nested(501), headers={"content-type": "application/json"})

    assert response.status_code == 400
    error = response.json()["detail"]["error"]
    assert (error["type"], error["message"]) == ("SchemaError", TOO_DEEP)


@pytest.mark.parametrize("path", ["/run", "/tactics/ok/run"])
def test_a_body_nested_500_levels_passes(path):
    client = TestClient(create_tactic_app(Ok()))

    response = client.post(path, content=_nested(500), headers={"content-type": "application/json"})

    assert response.json()["output"] == "ok"


def test_brackets_inside_strings_do_not_count():
    client = TestClient(create_tactic_app(Echo()))
    text = 'a\\"' + "[{" * 600  # an escaped quote does not end the string
    body = json.dumps({"input": json.loads(f'"{text}"')})

    response = client.post("/run", content=body, headers={"content-type": "application/json"})

    assert response.json()["output"] == json.loads(f'"{text}"')


def test_unbalanced_brackets_that_are_not_json_are_a_400_too():
    client = TestClient(create_tactic_app(Echo()), raise_server_exceptions=False)

    response = client.post("/run", content=b"[" * 20000, headers={"content-type": "text/plain"})

    assert response.status_code == 400
    assert response.json()["detail"]["error"]["message"] == TOO_DEEP


def test_a_recursion_error_while_parsing_is_the_same_400(monkeypatch):
    import types

    import lllm.services.fastapi as service

    def too_deep(text):
        raise RecursionError("maximum recursion depth exceeded")

    monkeypatch.setattr(service, "json", types.SimpleNamespace(loads=too_deep))

    with pytest.raises(SchemaError, match=TOO_DEEP):
        service._body_value(b"[1]", "application/json")


def test_the_depth_scan_stops_where_json_stops_reading():
    # As in AAAX: json.loads recurses only into a first value that is an
    # array or an object, and refuses anything after that value.
    client = TestClient(create_tactic_app(Echo()))
    deep = "[" * 20000
    as_text = [
        "1 " + deep,
        '"x" ' + deep,
        "[1] " + deep,
        "{} " + deep,
        # A first value that never closes and never passes 500 levels.
        json.dumps({"items": [{"id": i} for i in range(600)]})[:-1],
        '{"note": "' + "[" * 600,
        # A form feed is not JSON whitespace: json.loads stops at it.
        "\x0c" + "[" * 501 + "]" * 501,
    ]
    for body in as_text:
        response = client.post("/run", content=body, headers={"content-type": "text/plain"})
        assert response.status_code == 200, body[:12]
        assert response.json()["output"] == body  # not JSON: the text
    # Bodies json.loads could parse, so only the scan refuses them: after a
    # byte order mark and JSON whitespace, past a bracket that closed, and
    # past a string that ends in an escaped backslash.
    refused = [
        "\ufeff \n\t" + "[" * 501 + "]" * 501,
        " \r\n" + '{"a": ' * 501 + "1" + "}" * 501,
        '{"b": [1], "a": ' + "[" * 600 + "]" * 600 + "}",
        '["\\\\", ' + "[" * 501 + "]" * 501 + "]",
    ]
    for body in refused:
        json.loads(body.removeprefix("\ufeff"))
        response = client.post("/run", content=body.encode(), headers={"content-type": "application/json"})
        assert response.status_code == 400, body[:12]
        assert response.json()["detail"]["error"]["message"] == TOO_DEEP


def test_bodies_the_scan_accepts_pass():
    # As in AAAX: each has more than 500 brackets, so the scan reads it.
    client = TestClient(create_tactic_app(Ok()))
    accepted = [
        '{"a": "' + "[" * 20000 + '"}',
        json.dumps({"items": [{"id": i} for i in range(600)]}),
        '{"b": [1], "a": ' + '{"a": ' * 499 + "1" + "}" * 499 + "}",  # 500 deep
        '{"a": "\\"' + "[" * 600 + '"}',  # an escaped quote keeps the string open
        '{"a": "x\\\\", "b": "' + "[" * 600 + '"}',  # an escaped backslash ends it
    ]
    for body in accepted:
        json.loads(body)
        response = client.post("/run", content=body, headers={"content-type": "application/json"})
        assert response.json()["output"] == "ok", body[:30]


def test_a_big_body_is_read_off_the_event_loop():
    # An envelope is validated off the loop too, which only the
    # thread-recording test below can see.
    app = create_tactic_app(Ok())
    big = "[" + "[]" * 5_000_000 + "]"  # 10 MB, scanned to the end

    async def scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            done: list[str] = []

            async def post_big():
                response = await client.post(
                    "/run", content=big, headers={"content-type": "application/json"}
                )
                done.append("big")
                return response

            async def get_health():
                await asyncio.sleep(0.2)  # the big body is being read by now
                response = await client.get("/health")
                done.append("health")
                return response

            responses = await asyncio.gather(post_big(), get_health())
            return done, responses

    done, (big_response, health) = asyncio.run(scenario())

    assert done == ["health", "big"]
    assert big_response.status_code == health.status_code == 200


@pytest.mark.parametrize("path", ["/run", "/tactics/ok/run"])
def test_the_envelope_and_context_are_built_off_the_event_loop(path, monkeypatch):
    # Validating an envelope deep-copies its input, about 2 s for 10 MB, and
    # building the CallContext copies its metadata, about 1 s per MB, so both
    # run in the worker thread with the parse. A timing test cannot see
    # them: the parse before them lets a concurrent request through.
    import lllm.services.fastapi as service

    where: list[tuple[str, str]] = []

    def recording(name, function):
        def record(*args, **kwargs):
            try:
                asyncio.get_running_loop()
                where.append((name, "event loop"))
            except RuntimeError:
                where.append((name, "worker thread"))
            return function(*args, **kwargs)

        return record

    validate = service.RunRequest.model_validate
    monkeypatch.setattr(service.RunRequest, "model_validate", recording("envelope", validate))
    monkeypatch.setattr(
        service,
        "call_context_from_data",
        recording("context", service.call_context_from_data),
    )
    body = {"input": [[], []], "context": {"metadata": {"note": "x"}}}

    response = TestClient(create_tactic_app(Ok())).post(path, json=body)

    assert response.json()["output"] == "ok"
    assert where == [("envelope", "worker thread"), ("context", "worker thread")]


# Round 8, 1: the boundary copies walk values with a stack, not recursion,
# so validating a deep envelope in a worker thread cannot overflow a small
# thread stack (on CPython 3.10 each Python call also uses C stack).


def _nested_list(depth: int) -> list:
    value: list = []
    for _ in range(depth):
        value = [value]
    return value


def test_the_boundary_copies_keep_their_semantics():
    from collections import OrderedDict, namedtuple

    from lllm.protocol._validation import (
        _copy_metadata_value,
        _copy_string_keyed_value,
        copy_boundary_value,
    )

    Pair = namedtuple("Pair", "a b")
    shared = [1]
    leaf = object()
    value = OrderedDict(
        a=[shared, shared], b=(1, Pair(2, 3)), c={4}, d=frozenset({5}), e=leaf
    )

    copied = copy_boundary_value(value)

    assert copied == {"a": [[1], [1]], "b": (1, (2, 3)), "c": {4}, "d": frozenset({5}), "e": copied["e"]}
    assert type(copied) is dict and type(copied["b"][1]) is tuple
    assert copied["a"][0] is not shared and copied["a"][0] is not copied["a"][1]
    assert copied["e"] is not leaf  # deep-copied
    for copy in (_copy_string_keyed_value, _copy_metadata_value):
        assert copy("context", {"a": [{"b": 1}]}) == {"a": [{"b": 1}]}
        with pytest.raises(TypeError, match=r"^context\.a keys must be strings\.$"):
            copy("context", {"a": [{1: "x"}]})
    cycle: list = []
    cycle.append(cycle)
    with pytest.raises(RecursionError):
        copy_boundary_value(cycle)


def test_the_boundary_copies_do_not_recurse():
    from lllm.protocol._validation import _copy_string_keyed_value, copy_boundary_value

    deep = _nested_list(50_000)

    copied = copy_boundary_value(deep)
    keyed = _copy_string_keyed_value("context", {"a": deep})

    for value in (copied, keyed["a"]):
        node, original = value, deep
        for _ in range(50_000):
            assert node is not original and len(node) == 1
            node, original = node[0], original[0]
        assert node == [] and node is not original


SMALL_STACK_ENVELOPE_SCRIPT = """
import sys
import threading

threading.stack_size(1 << 20)  # 1 MiB thread stacks, as python:alpine builds use
# Python 3.12 runs Python calls without C stack, so a low limit stands in for
# the frames per level a recursive copy spends. On 3.10 and 3.11 the limit
# also caps json.loads and list comparison, and the real 1 MiB stack and the
# default limit already break a recursive copy.
if sys.version_info >= (3, 12):
    sys.setrecursionlimit(400)

from fastapi.testclient import TestClient

from lllm.protocol import Tactic
from lllm.protocol._validation import copy_boundary_value
from lllm.services import create_tactic_app

received = []


class Keep(Tactic):
    name = "keep"

    def _run(self, value, **kwargs):
        received.append(value)
        return "kept"


depth = 499  # with the envelope, 500 levels
body = '{"input": ' + "[" * depth + "]" * depth + ', "context": {"metadata": {"a": ' + "[" * 400 + "]" * 400 + "}}}"
client = TestClient(create_tactic_app(Keep()), raise_server_exceptions=False)
response = client.post("/run", content=body, headers={"content-type": "application/json"})
print("status", response.status_code, response.json().get("output"), flush=True)

expected = []
for _ in range(depth - 1):
    expected = [expected]
print("equal", received[0] == expected, flush=True)

results = []


def copy_in_thread():
    copied = copy_boundary_value(received[0])
    node, original = copied, received[0]
    independent = True
    while node:
        independent = independent and node is not original
        node, original = node[0], original[0]
    results.append(copied == received[0] and independent)


thread = threading.Thread(target=copy_in_thread)
thread.start()
thread.join()
print("copy", results, flush=True)
"""


def test_a_deep_envelope_is_validated_on_a_small_thread_stack(tmp_path):
    import lllm

    script = tmp_path / "envelope.py"
    script.write_text(SMALL_STACK_ENVELOPE_SCRIPT, encoding="utf-8")
    root = str(Path(lllm.__file__).resolve().parents[1])
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([root, os.environ.get("PYTHONPATH", "")])}

    result = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, env=env, timeout=120
    )

    assert result.returncode == 0, (result.stdout, result.stderr[-1500:])
    assert result.stdout.split("\n")[:3] == ["status 200 kept", "equal True", "copy [True]"]


DEEP_BODY_SCRIPT = """
import threading

threading.stack_size(1 << 20)  # 1 MiB thread stacks, as python:alpine builds use

from fastapi.testclient import TestClient

from lllm import Tactic, create_tactic_app


class Deep(Tactic):
    name = "deep"

    def call(self, task: dict) -> str:
        return "ok"


client = TestClient(create_tactic_app(Deep()), raise_server_exceptions=False)
for depth in (4500, 6000):
    body = '{"a": ' * depth + "1" + "}" * depth  # a bare body, not an envelope
    response = client.post("/run", content=body, headers={"content-type": "application/json"})
    print(depth, response.status_code, flush=True)
print("alive", client.post("/run", json={"input": {"a": 1}}).json()["output"], flush=True)
"""


def test_a_deep_body_cannot_crash_a_server_with_small_thread_stacks(tmp_path):
    import lllm

    script = tmp_path / "deep.py"
    script.write_text(DEEP_BODY_SCRIPT, encoding="utf-8")
    root = str(Path(lllm.__file__).resolve().parents[1])
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([root, os.environ.get("PYTHONPATH", "")])}

    result = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, env=env, timeout=120
    )

    assert result.returncode == 0, (result.returncode, result.stderr[-1500:])
    assert result.stdout.split("\n")[:3] == ["4500 400", "6000 400", "alive ok"]


def test_an_octet_stream_body_that_is_not_utf8_reaches_the_tactic_as_bytes():
    class Size(ProtocolTactic):
        name = "size"

        def _run(self, value, **kwargs):
            return {"type": type(value).__name__, "size": len(value)}

    client = TestClient(create_tactic_app(Size()))
    headers = {"content-type": "application/octet-stream"}

    binary = client.post("/run", content=b"\xff\xfe\x00\x01", headers=headers)
    text = client.post("/run", content=b"hi", headers=headers)

    assert binary.json()["output"] == {"type": "bytes", "size": 4}
    assert text.json()["output"] == {"type": "str", "size": 2}


@pytest.mark.parametrize("path", ["/run", "/tactics/echo/run"])
def test_a_tactic_that_echoes_octet_stream_bytes_gets_a_500_envelope(path):
    client = TestClient(create_tactic_app(Echo()), raise_server_exceptions=False)

    response = client.post(
        path, content=b"\xff\xfe", headers={"content-type": "application/octet-stream"}
    )

    assert response.status_code == 500
    assert response.json()["detail"]["error"]["type"] == "UnicodeDecodeError"


def test_utf8_text_and_json_bodies_still_reach_the_tactic():
    client = TestClient(create_tactic_app(Echo()))

    text = client.post(
        "/run", content="héllo".encode(), headers={"content-type": "text/plain"}
    )
    with_bom = client.post(
        "/run",
        content=b'\xef\xbb\xbf{"input": "hi"}',
        headers={"content-type": "application/json"},
    )

    assert text.json()["output"] == "héllo"
    assert with_bom.json()["output"] == "hi"


class Unencodable(ProtocolTactic):
    name = "unencodable"

    def _run(self, value, **kwargs):
        return {"bytes": b"\xff", "surrogate": "\ud800"}[value]


@pytest.mark.parametrize("path", ["/run", "/tactics/unencodable/run"])
@pytest.mark.parametrize("kind", ["bytes", "surrogate"])
def test_a_result_that_cannot_be_encoded_is_a_500_in_the_envelope(path, kind):
    client = TestClient(create_tactic_app(Unencodable()), raise_server_exceptions=False)

    response = client.post(path, json={"input": kind, "context": {"request_id": "r-7"}})

    assert response.status_code == 500
    error = response.json()["detail"]["error"]
    assert error["tactic"] == "unencodable"
    assert error["request_id"] == "r-7"


def test_a_custom_endpoint_result_that_cannot_be_encoded_is_a_500_in_the_envelope():
    class Shout(ProtocolTactic):
        name = "echo"

        def _run(self, value, **kwargs):
            return value

        @endpoint.post("/echo/raw")
        def raw(self, value):
            return "\ud800"

    client = TestClient(create_tactic_app(Shout()), raise_server_exceptions=False)

    response = client.post("/echo/raw", json={"input": "x"})

    assert response.status_code == 500
    assert response.json()["detail"]["error"]["endpoint"] == "raw"


# Review 8: with --no-keyring, lllm serve stores the keys it asks for in the
# env file, where its own check looks, never in the OS keyring.


@pytest.fixture
def key_package(tmp_path, monkeypatch):
    """A package that needs FIX3_TEST_KEY, served from a fake terminal."""

    import types

    package = tmp_path / "keyed"
    _write(
        package / "lllm.toml",
        """
[package]
org = "demo"
name = "keyed"
primary = "tactics.echo"

[requirements.api_keys]
FIX3_TEST_KEY = "Model access."

[tactics.echo]
entry = "fix3_keyed:Echo"
""",
    )
    _write(
        package / "fix3_keyed.py",
        "class Echo:\n    def run(self, value, *, context=None):\n        return value\n",
    )
    monkeypatch.delenv("FIX3_TEST_KEY", raising=False)
    monkeypatch.setattr(sys, "stdin", types.SimpleNamespace(isatty=lambda: True))
    return package


def _type_keys(monkeypatch, answer) -> None:
    """Answer key prompts with ``answer``, or raise it if it is an exception."""

    import lllm.credentials

    def secret(prompt):
        if isinstance(answer, BaseException):
            raise answer
        return answer

    typed = functools.partial(
        lllm.credentials.configure_missing_api_keys, secret_func=secret
    )
    monkeypatch.setattr(lllm.credentials, "configure_missing_api_keys", typed)


@pytest.fixture
def keyed_package(key_package, monkeypatch):
    import lllm.credentials

    # A fake keyring: it looks available, and keeps what is stored in it here
    # instead of in the OS keyring.
    monkeypatch.setattr(lllm.credentials, "keyring_available", lambda: True)
    stored: dict[str, str] = {}
    monkeypatch.setattr(lllm.credentials, "keyring_set", stored.__setitem__)
    monkeypatch.setattr(lllm.credentials, "keyring_get", stored.get)
    _type_keys(monkeypatch, "sk-typed")
    return key_package, stored


@pytest.fixture
def fail_keyring():
    """The real keyring module with its fail backend, as on headless hosts."""

    keyring = pytest.importorskip("keyring")
    from keyring.backends import fail

    previous = keyring.get_keyring()
    keyring.set_keyring(fail.Keyring())
    yield
    keyring.set_keyring(previous)


def test_keyring_available_needs_a_usable_backend(fail_keyring, monkeypatch):
    import types

    import keyring
    import keyring.backend
    from keyring.backends import chainer

    from lllm import credentials

    assert not credentials.keyring_available()  # the fail backend
    assert credentials.keyring_get("FIX3_TEST_KEY") is None  # not an error
    for priority, usable in ((5, True), (0, False), (-1, False)):
        backend = types.SimpleNamespace(priority=priority)
        monkeypatch.setattr(keyring, "get_keyring", lambda backend=backend: backend)
        assert credentials.keyring_available() is usable
    # A chainer with nothing to chain, as get_keyring() may return.
    monkeypatch.setattr(keyring.backend, "get_all_keyring", lambda: [])
    monkeypatch.setattr(keyring, "get_keyring", chainer.ChainerBackend)
    assert not credentials.keyring_available()


def test_choose_credential_backend_stays_a_pure_mapping(fail_keyring):
    from lllm.credentials import CredentialError, choose_credential_backend, store_api_keys

    # As in 0.4.0, which AAAX's CLI calls directly.
    assert choose_credential_backend("keyring") == "keyring"
    assert choose_credential_backend("env") == "env"
    assert choose_credential_backend("auto") == "env"  # no usable keyring
    with pytest.raises(CredentialError, match="no usable OS keyring"):
        store_api_keys({"FIX3_TEST_KEY": "sk"}, credentials="keyring")


def test_auto_falls_back_to_the_env_file_without_a_usable_keyring(
    key_package, fail_keyring, served, tmp_path, monkeypatch, capsys
):
    _type_keys(monkeypatch, "sk-typed")
    env_file = tmp_path / "keys.env"

    assert main(["serve", str(key_package), "--env-file", str(env_file)]) == 0

    assert 'FIX3_TEST_KEY="sk-typed"' in env_file.read_text(encoding="utf-8")
    assert f"via {env_file}" in capsys.readouterr().out
    assert len(served) == 1


@pytest.mark.parametrize("command", ["serve", "setup"])
def test_keyring_credentials_without_a_usable_keyring_stop_before_asking(
    key_package, fail_keyring, served, tmp_path, monkeypatch, capsys, command
):
    _type_keys(monkeypatch, AssertionError("asked for keys before refusing"))
    env_file = tmp_path / "keys.env"

    with pytest.raises(SystemExit) as caught:
        main([command, str(key_package), "--credentials", "keyring", "--env-file", str(env_file)])

    assert caught.value.code == 2
    assert "no usable OS keyring" in capsys.readouterr().err
    assert not env_file.exists() and served == []


def test_serve_no_keyring_stores_prompted_keys_in_the_env_file(
    keyed_package, served, tmp_path, capsys
):
    package, stored = keyed_package
    env_file = tmp_path / "keys.env"

    args = ["serve", str(package), "--no-keyring", "--env-file", str(env_file)]
    assert main(args) == 0

    assert stored == {}
    assert 'FIX3_TEST_KEY="sk-typed"' in env_file.read_text(encoding="utf-8")
    assert f"via {env_file}" in capsys.readouterr().out
    assert len(served) == 1


def test_serve_refuses_no_keyring_with_keyring_credentials(
    keyed_package, served, capsys
):
    package, stored = keyed_package

    with pytest.raises(SystemExit) as caught:
        main(["serve", str(package), "--no-keyring", "--credentials", "keyring"])

    assert caught.value.code == 2
    err = capsys.readouterr().err
    assert "--no-keyring" in err and "--credentials keyring" in err
    assert stored == {} and served == []


def test_serve_allows_no_keyring_with_keyring_credentials_when_no_key_check_runs(
    keyed_package, app_module, served
):
    package, stored = keyed_package
    contradictory = ["--no-keyring", "--credentials", "keyring"]

    assert main(["serve", f"{app_module}:shout", *contradictory]) == 0
    assert main(["serve", str(package), *contradictory, "--skip-key-check"]) == 0

    assert len(served) == 2 and stored == {}


def test_serve_still_stores_prompted_keys_through_credentials(
    keyed_package, served, tmp_path, capsys
):
    package, stored = keyed_package
    env_file = tmp_path / "keys.env"

    assert main(["serve", str(package), "--env-file", str(env_file)]) == 0

    assert stored == {"FIX3_TEST_KEY": "sk-typed"}
    assert not env_file.exists()
    assert "via keyring" in capsys.readouterr().out
    assert len(served) == 1


# Round 4, 7: an env file that is a symlink, dangling or looping, is refused
# before any key is asked for, so typed keys are never written through it.


def _symlinked_env_file(tmp_path: Path, kind: str) -> tuple[Path, Path | None]:
    env_file = tmp_path / ".env.local"
    if kind == "dangling":
        target = tmp_path / "elsewhere.env"  # its folder exists, the file not
        env_file.symlink_to(target)
        return env_file, target
    env_file.symlink_to(env_file)  # a link to itself
    return env_file, None


@pytest.mark.parametrize("kind", ["dangling", "looping"])
@pytest.mark.parametrize("command", ["serve", "setup"])
def test_an_env_file_that_is_a_symlink_is_refused(
    keyed_package, served, tmp_path, monkeypatch, capsys, kind, command
):
    import lllm.credentials

    package, stored = keyed_package
    env_file, target = _symlinked_env_file(tmp_path, kind)
    args = [command, str(package), "--credentials", "env", "--env-file", str(env_file)]
    if command == "setup":
        args += ["--set", "FIX3_TEST_KEY=sk-set"]

    def asked(*args, **kwargs):
        raise AssertionError("asked for keys before refusing")

    monkeypatch.setattr(lllm.credentials, "configure_missing_api_keys", asked)

    with pytest.raises(SystemExit) as caught:
        main(args)

    assert caught.value.code == 2
    assert "env file must be a regular file" in capsys.readouterr().err
    assert env_file.is_symlink()
    assert target is None or not target.exists()
    assert served == [] and stored == {}


@pytest.mark.parametrize("kind", ["dangling", "looping"])
def test_the_env_file_helpers_refuse_a_symlink(tmp_path, kind):
    from lllm.credentials import CredentialError, read_env_file, write_env_file

    env_file, target = _symlinked_env_file(tmp_path, kind)

    with pytest.raises(CredentialError, match="env file must be a regular file"):
        read_env_file(env_file)
    with pytest.raises(CredentialError, match="env file must be a regular file"):
        write_env_file(env_file, {"FIX3_TEST_KEY": "sk"})
    assert target is None or not target.exists()


# Round 5, 3: an env file that cannot be written is a clean error, found
# before keys are asked for when possible, and written without following a
# symlink swapped in after the check.


def _unwritable_env_file(tmp_path: Path, kind: str) -> Path:
    if kind == "read-only-folder":
        folder = tmp_path / "read-only"
        folder.mkdir()
        folder.chmod(0o500)
        return folder / "keys.env"
    blocker = tmp_path / "blocker"  # a file where a folder should be
    blocker.write_text("not a folder\n", encoding="utf-8")
    return blocker / "keys.env"


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root writes anywhere")
@pytest.mark.parametrize("kind", ["read-only-folder", "parent-is-a-file"])
@pytest.mark.parametrize("command", ["serve", "setup"])
def test_an_unwritable_env_file_stops_before_asking(
    key_package, served, tmp_path, monkeypatch, capsys, kind, command
):
    _type_keys(monkeypatch, AssertionError("asked for keys before refusing"))
    env_file = _unwritable_env_file(tmp_path, kind)
    args = [command, str(key_package), "--credentials", "env", "--env-file", str(env_file)]
    try:
        with pytest.raises(SystemExit) as caught:
            main(args)
    finally:
        if kind == "read-only-folder":
            env_file.parent.chmod(0o700)

    assert caught.value.code == 2
    err = capsys.readouterr().err
    assert f"cannot write env file {env_file}" in err
    assert "Traceback" not in err
    assert served == [] and not env_file.exists()


@pytest.mark.parametrize("env_file", ["~fix3-no-such-user/keys.env", "./~keys.env"])
@pytest.mark.parametrize("command", ["serve", "setup"])
def test_an_env_file_path_that_cannot_be_expanded_is_a_clean_error(
    key_package, served, tmp_path, monkeypatch, capsys, command, env_file
):
    _type_keys(monkeypatch, AssertionError("asked for keys before refusing"))
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as caught:
        main([command, str(key_package), "--env-file", env_file])

    assert caught.value.code == 2
    err = capsys.readouterr().err
    assert f"cannot expand env file {env_file}: Could not determine home directory" in err
    assert "Traceback" not in err and served == []


@pytest.mark.parametrize("target", ["~fix3-no-such-user/pkg", "./~pkg"])
@pytest.mark.parametrize("command", ["serve", "inspect", "setup"])
def test_a_target_that_cannot_be_expanded_is_a_clean_error(
    served, tmp_path, monkeypatch, capsys, command, target
):
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as caught:
        main([command, target])

    assert caught.value.code == 2
    err = capsys.readouterr().err
    assert f"cannot expand {target}: Could not determine home directory" in err
    assert "Traceback" not in err and served == []


def test_the_launch_api_refuses_a_target_it_cannot_expand():
    from lllm.launch import LaunchError, load_launch_api_key_requirements, load_launch_app

    for call in (load_launch_app, load_launch_api_key_requirements):
        with pytest.raises(LaunchError, match="cannot expand ~fix3-no-such-user/pkg: "):
            call("~fix3-no-such-user/pkg")


@pytest.mark.parametrize(
    ("args", "path"),
    [
        (["init", "~fix3-no-such-user/pkg"], "~fix3-no-such-user/pkg"),
        (["hub", "--hub", "~fix3-no-such-user/hub", "list"], "~fix3-no-such-user/hub"),
    ],
    ids=["init", "hub-root"],
)
def test_init_and_the_hub_refuse_a_path_they_cannot_expand(capsys, args, path):
    with pytest.raises(SystemExit) as caught:
        main(args)

    assert caught.value.code == 2
    err = capsys.readouterr().err
    assert f"cannot expand {path}: Could not determine home directory" in err
    assert "Traceback" not in err


@pytest.mark.parametrize("command", ["serve", "inspect"])
@pytest.mark.parametrize("kind", ["unreadable", "too-long"])
def test_a_target_that_cannot_be_read_is_a_clean_error(
    served, tmp_path, capsys, command, kind
):
    if kind == "unreadable":
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            pytest.skip("root reads anywhere")
        locked = tmp_path / "locked"
        locked.mkdir()
        target, reason = str(locked / "pkg"), "Permission denied"
        locked.chmod(0o000)
    else:
        target, reason = str(tmp_path / ("x" * 300)), "File name too long"
    try:
        with pytest.raises(SystemExit) as caught:
            main([command, target])
    finally:
        if kind == "unreadable":
            locked.chmod(0o700)

    assert caught.value.code == 2
    err = capsys.readouterr().err
    assert f"cannot read {target}: {reason}" in err
    assert "Traceback" not in err and served == []


class _FailingOs:
    """``os`` for lllm.credentials, with chosen functions replaced."""

    def __init__(self, **replacements):
        self._replacements = replacements

    def __getattr__(self, name):
        return self._replacements.get(name) or getattr(os, name)


def test_a_failed_write_is_a_clean_error_after_keys_are_typed(
    key_package, served, tmp_path, monkeypatch, capsys
):
    import errno

    import lllm.credentials

    def no_space(*args, **kwargs):
        raise OSError(errno.ENOSPC, "No space left on device")

    _type_keys(monkeypatch, "sk-typed")
    monkeypatch.setattr(lllm.credentials, "os", _FailingOs(open=no_space))
    env_file = tmp_path / "keys.env"

    with pytest.raises(SystemExit) as caught:
        main(["serve", str(key_package), "--credentials", "env", "--env-file", str(env_file)])

    assert caught.value.code == 2
    err = capsys.readouterr().err
    assert f"cannot write env file {env_file}: No space left on device" in err
    assert served == []


def test_a_failed_fchmod_keeps_every_key_in_the_env_file(tmp_path, monkeypatch):
    import errno

    import lllm.credentials
    from lllm.credentials import write_env_file

    def not_the_owner(fd, mode):  # a group-writable file, or a FAT/CIFS mount
        raise OSError(errno.EPERM, "Operation not permitted")

    env_file = tmp_path / ".env.local"
    env_file.write_text('OTHER_KEY="keep-me"\n', encoding="utf-8")
    monkeypatch.setattr(lllm.credentials, "os", _FailingOs(fchmod=not_the_owner))

    write_env_file(env_file, {"FIX3_TEST_KEY": "sk"})

    text = env_file.read_text(encoding="utf-8")
    assert 'OTHER_KEY="keep-me"' in text
    assert 'FIX3_TEST_KEY="sk"' in text


def test_any_os_error_from_the_env_file_helpers_is_a_credential_error(tmp_path):
    from lllm.credentials import CredentialError, check_env_file_writable, write_env_file

    too_long = tmp_path / ("x" * 300)  # longer than a file name may be

    with pytest.raises(CredentialError, match="cannot write env file .*: File name too long"):
        write_env_file(too_long, {"FIX3_TEST_KEY": "sk"})
    with pytest.raises(CredentialError, match="cannot write env file .*: File name too long"):
        check_env_file_writable(too_long)


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root reads anywhere")
def test_an_unreadable_env_file_folder_is_a_credential_error_everywhere(tmp_path):
    # On Python 3.14 pathlib's exists() and is_symlink() return False for
    # EACCES instead of raising, so the helpers probe with os.lstat.
    from lllm.credentials import (
        CredentialError,
        check_env_file_writable,
        read_env_file,
        write_env_file,
    )

    folder = tmp_path / "locked"
    folder.mkdir()
    env_file = folder / "keys.env"
    folder.chmod(0o000)
    try:
        with pytest.raises(CredentialError, match="cannot read env file .*: Permission denied"):
            read_env_file(env_file)
        with pytest.raises(CredentialError, match="cannot write env file .*: Permission denied"):
            check_env_file_writable(env_file)
        with pytest.raises(CredentialError, match="cannot write env file .*: Permission denied"):
            write_env_file(env_file, {"FIX3_TEST_KEY": "sk"})
    finally:
        folder.chmod(0o700)


def test_an_env_file_that_is_not_utf8_is_a_credential_error(
    key_package, served, tmp_path, monkeypatch, capsys
):
    from lllm.credentials import CredentialError, read_env_file, write_env_file

    env_file = tmp_path / ".env.local"
    env_file.write_bytes(b'OTHER="caf\xe9"\n')  # Latin-1, not UTF-8
    message = f"env file {env_file} is not UTF-8 text"

    with pytest.raises(CredentialError, match=message):
        read_env_file(env_file)
    with pytest.raises(CredentialError, match=message):
        write_env_file(env_file, {"FIX3_TEST_KEY": "sk"})
    _type_keys(monkeypatch, AssertionError("asked for keys before refusing"))
    with pytest.raises(SystemExit) as caught:
        main(["serve", str(key_package), "--credentials", "env", "--env-file", str(env_file)])
    assert caught.value.code == 2
    assert message in capsys.readouterr().err
    assert env_file.read_bytes() == b'OTHER="caf\xe9"\n' and served == []


def test_a_file_someone_else_owns_keeps_its_permissions(tmp_path, monkeypatch):
    # Root (or CAP_FOWNER) could fchmod a file it does not own, so the
    # owner is checked rather than the call left to fail.
    import lllm.credentials
    from lllm.credentials import write_env_file

    changed = []
    env_file = tmp_path / ".env.local"
    env_file.write_text("OTHER=1\n", encoding="utf-8")
    env_file.chmod(0o664)
    monkeypatch.setattr(
        lllm.credentials,
        "os",
        _FailingOs(
            geteuid=lambda: os.stat(env_file).st_uid + 1,  # not the owner
            fchmod=lambda fd, mode: changed.append(mode),
        ),
    )

    write_env_file(env_file, {"FIX3_TEST_KEY": "sk"})

    assert changed == []
    assert env_file.stat().st_mode & 0o777 == 0o664
    assert env_file.read_text(encoding="utf-8") == 'OTHER=1\n\nFIX3_TEST_KEY="sk"\n'


def test_a_rewrite_that_shortens_the_env_file_leaves_no_old_bytes(tmp_path):
    from lllm.credentials import write_env_file

    env_file = tmp_path / ".env.local"
    env_file.write_text(
        'FIX3_TEST_KEY="sk-a-much-longer-old-value"\nOTHER=1\n', encoding="utf-8"
    )

    write_env_file(env_file, {"FIX3_TEST_KEY": "sk"})

    assert env_file.read_text(encoding="utf-8") == 'FIX3_TEST_KEY="sk"\nOTHER=1\n'


def test_an_existing_env_file_is_made_private_and_keeps_its_keys(tmp_path):
    from lllm.credentials import write_env_file

    env_file = tmp_path / ".env.local"
    env_file.write_text("OTHER=1\n", encoding="utf-8")
    env_file.chmod(0o644)

    write_env_file(env_file, {"FIX3_TEST_KEY": "sk"})

    assert env_file.stat().st_mode & 0o777 == 0o600
    assert env_file.read_text(encoding="utf-8") == 'OTHER=1\n\nFIX3_TEST_KEY="sk"\n'


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root writes anywhere")
def test_a_read_only_env_file_stops_before_asking(
    key_package, served, tmp_path, monkeypatch, capsys
):
    _type_keys(monkeypatch, AssertionError("asked for keys before refusing"))
    env_file = tmp_path / ".env.local"
    env_file.write_text("OTHER=1\n", encoding="utf-8")
    env_file.chmod(0o400)

    try:
        with pytest.raises(SystemExit) as caught:
            main(["serve", str(key_package), "--credentials", "env", "--env-file", str(env_file)])
    finally:
        env_file.chmod(0o600)

    assert caught.value.code == 2
    assert f"cannot write env file {env_file}" in capsys.readouterr().err
    assert env_file.read_text(encoding="utf-8") == "OTHER=1\n" and served == []


def test_an_env_file_in_missing_folders_is_created_private(
    key_package, served, tmp_path, monkeypatch
):
    _type_keys(monkeypatch, "sk-typed")
    env_file = tmp_path / "new" / "sub" / "keys.env"

    assert main(["serve", str(key_package), "--credentials", "env", "--env-file", str(env_file)]) == 0

    assert env_file.read_text(encoding="utf-8") == 'FIX3_TEST_KEY="sk-typed"\n'
    assert env_file.stat().st_mode & 0o777 == 0o600
    assert len(served) == 1


@pytest.mark.parametrize("kind", ["dangling", "looping"])
def test_an_env_file_under_a_broken_folder_link_stops_before_asking(
    key_package, served, tmp_path, monkeypatch, capsys, kind
):
    _type_keys(monkeypatch, AssertionError("asked for keys before refusing"))
    folder = tmp_path / "linked"
    folder.symlink_to(tmp_path / "missing" if kind == "dangling" else folder)
    env_file = folder / "keys.env"

    with pytest.raises(SystemExit) as caught:
        main(["serve", str(key_package), "--credentials", "env", "--env-file", str(env_file)])

    assert caught.value.code == 2
    expected = (
        f"cannot write env file {env_file}: {folder} is not a folder"
        if kind == "dangling"
        # Reading the file through a looping link already fails.
        else f"cannot read env file {env_file}: Too many levels of symbolic links"
    )
    assert expected in capsys.readouterr().err
    assert served == [] and not (tmp_path / "missing").exists()


def test_write_env_file_opens_without_following_links(tmp_path, monkeypatch):
    import lllm.credentials
    from lllm.credentials import write_env_file

    flags_seen = []

    def recording_open(path, flags, mode=0o777):
        flags_seen.append(flags)
        return os.open(path, flags, mode)

    def no_chmod(self, mode):
        raise AssertionError("Path.chmod follows links; use os.fchmod")

    monkeypatch.setattr(lllm.credentials, "os", _FailingOs(open=recording_open))
    monkeypatch.setattr(Path, "chmod", no_chmod)
    env_file = tmp_path / "keys.env"

    write_env_file(env_file, {"FIX3_TEST_KEY": "sk"})

    assert flags_seen and flags_seen[0] & getattr(os, "O_NOFOLLOW", 0)
    assert not flags_seen[0] & os.O_TRUNC  # emptied only once the checks pass
    assert env_file.stat().st_mode & 0o777 == 0o600
    assert 'FIX3_TEST_KEY="sk"' in env_file.read_text(encoding="utf-8")


def test_write_env_file_refuses_a_symlink_swapped_in_after_the_check(
    tmp_path, monkeypatch
):
    from lllm.credentials import CredentialError, write_env_file

    target = tmp_path / "elsewhere.env"
    target.write_text("KEEP=1\n", encoding="utf-8")
    env_file = tmp_path / "keys.env"
    env_file.symlink_to(target)
    # The pre-check sees no symlink, as if the link appeared just after it.
    monkeypatch.setattr(Path, "is_symlink", lambda self: False)

    with pytest.raises(CredentialError, match="env file must be a regular file"):
        write_env_file(env_file, {"FIX3_TEST_KEY": "sk"})

    assert target.read_text(encoding="utf-8") == "KEEP=1\n"


def test_write_env_file_refuses_a_file_that_is_not_regular_once_open(
    tmp_path, monkeypatch
):
    import stat as stat_module

    import lllm.credentials
    from lllm.credentials import CredentialError, write_env_file

    def fifo_stat(fd):
        real = os.fstat(fd)
        mode = stat_module.S_IFIFO | (real.st_mode & 0o777)
        return os.stat_result((mode, *tuple(real)[1:]))

    monkeypatch.setattr(lllm.credentials, "os", _FailingOs(fstat=fifo_stat))

    with pytest.raises(CredentialError, match="env file must be a regular file"):
        write_env_file(tmp_path / "keys.env", {"FIX3_TEST_KEY": "sk"})


def _help(command: str, monkeypatch, capsys) -> str:
    """``lllm COMMAND --help`` on one line.

    argparse wraps to the terminal width and breaks at hyphens, so the width
    is pinned. Color is off for every test (tests/conftest.py).
    """

    monkeypatch.setenv("COLUMNS", "1000")
    with pytest.raises(SystemExit):
        main([command, "--help"])
    return " ".join(capsys.readouterr().out.split())


def test_serve_help_says_where_typed_keys_go(monkeypatch, capsys):
    serve_help = _help("serve", monkeypatch, capsys)
    inspect_help = _help("inspect", monkeypatch, capsys)

    assert (
        "--env-file ENV_FILE Env file to read API keys from; typed keys are stored "
        "here with --credentials env, with --no-keyring, or with auto when no "
        "usable keyring is available"
    ) in serve_help
    assert (
        "--no-keyring Neither read API keys from the OS keyring nor store typed "
        "keys there; they go to --env-file"
    ) in serve_help
    assert (
        "auto (the default) picks the OS keyring when a usable one is available "
        "and the env file otherwise; --no-keyring makes it the env file, and "
        "refuses --credentials keyring"
    ) in serve_help
    assert "--env-file ENV_FILE Env file to read API keys from" in inspect_help
    assert "--no-keyring Do not read API keys from the OS keyring" in inspect_help
    assert "typed keys" not in inspect_help


def test_setup_help_says_where_keys_go(monkeypatch, capsys):
    setup_help = _help("setup", monkeypatch, capsys)

    assert (
        "--credentials {auto,keyring,env} Where keys are stored: auto (the "
        "default) picks a usable OS keyring when there is one, and the env file "
        "otherwise"
    ) in setup_help
    assert (
        "--env-file ENV_FILE Env file to read API keys from; keys are stored "
        "here with --credentials env, or with auto when no usable keyring is "
        "available"
    ) in setup_help


def test_status_code_for_error_maps_cross_site_requests_to_403():
    from lllm.services import CrossSiteRequestError

    assert status_code_for_error(CrossSiteRequestError("refused")) == 403


def test_inspecting_a_channel_package_creates_nothing_with_real_sssn(
    channel_package, capsys
):
    pytest.importorskip("sssn", reason="needs the sssn package")

    assert main(["inspect", str(channel_package)]) == 0

    assert "POST /channels" in capsys.readouterr().out
    assert not (channel_package / ".lllm").exists()
