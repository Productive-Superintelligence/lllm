"""Regression tests for serving, the CLI, RemoteTactic, and the hub (fix3)."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Literal

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel, ValidationError

from lllm import RemoteTactic, RemoteTacticError, SchemaError
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


def test_load_tactic_entrypoint_refuses_an_app(app_module):
    with pytest.raises(TypeError, match="app, not a tactic"):
        load_tactic_entrypoint(f"{app_module}:app")


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


def test_local_types_and_a_given_contract_win_over_nothing():
    contract = Triage().info()
    remote = RemoteTactic(
        "http://triage.internal:8000",
        output_type=int,
        contract=contract.model_dump(mode="json"),
    )

    info = remote.info()
    assert remote.tactic_name == "triage.internal-8000"
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

    sync_planner = Agent(model, tools=[tactic_as_tool(changelog)])
    with pytest.raises(UserError, match="synchronous tool"):
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
        "services.api": 9000,
        "services.plain": 8000,
        "services.multi": 8000,
        "services.bundle": 9300,
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
