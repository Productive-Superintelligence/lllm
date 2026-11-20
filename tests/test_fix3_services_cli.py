"""Regression tests for serving, the CLI, RemoteTactic, and the hub (fix3)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from lllm.cli import load_tactic_entrypoint, main


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
