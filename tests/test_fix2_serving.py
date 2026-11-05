"""Regression tests for serving packages and entrypoints (fix2 serving group)."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from lllm import Tactic as NativeTactic
from lllm.cli import main
from lllm.hub import import_entrypoint, validate_package
from lllm.launch import LaunchError, load_launch_api_key_requirements, load_launch_app
from lllm.protocol import Tactic
from lllm.runtimes import coerce_tactic
from lllm.runtimes.python import CallableTactic
from lllm.shell import create_strategy_app, load_strategy


@pytest.fixture(autouse=True)
def _forget_served_modules(monkeypatch):
    """Serving keeps package modules and paths loaded; drop this file's afterwards."""

    monkeypatch.setattr(sys, "path", list(sys.path))
    yield
    for name in list(sys.modules):
        if name.split(".", 1)[0].startswith("fix2_"):
            del sys.modules[name]


# 1: a service entry that fails to import or build never falls back to the
# bare tactic, which would serve it without the service's auth.


@pytest.mark.parametrize(
    ("service_source", "cause"),
    [
        ("import os\nTOKEN = os.environ['FIX2_MISSING_TOKEN']\n", KeyError),
        ("def create_app():\n    raise RuntimeError('no auth config')\n", RuntimeError),
        ("def create_app(settings):\n    return settings\n", None),
    ],
    ids=["import-fails", "factory-raises", "cannot-build"],
)
def test_a_service_entry_that_fails_is_an_error_not_the_bare_tactic(
    tmp_path: Path, service_source: str, cause: type[Exception] | None, capsys
):
    package = _package(
        tmp_path,
        primary="services.api",
        body="""
[tactics.echo]
entry = "fix2_svc.tactics:Echo"

[services.api]
entry = "fix2_svc.app:create_app"
tactic = "echo"
""",
    )
    _write(package / "fix2_svc" / "tactics.py", _ECHO)
    _write(package / "fix2_svc" / "app.py", service_source)

    with pytest.raises(LaunchError) as raised:
        load_launch_app(package)

    message = str(raised.value)
    if cause is None:
        assert message == "services.api did not return an ASGI app or service payload"
    else:
        assert isinstance(raised.value.__cause__, cause)
        assert message.startswith(
            f"service api entry fix2_svc.app:create_app failed: {cause.__name__}"
        )
    with pytest.raises(SystemExit) as exited:
        main(["inspect", str(package), "--no-keyring"])
    assert exited.value.code == 2
    assert f"lllm: error: {message}" in capsys.readouterr().err


# A service entry must build an app, a tactic, or a data payload; only a data
# payload lets a declared tactic be served instead.


@pytest.mark.filterwarnings("ignore:coroutine .* was never awaited:RuntimeWarning")
@pytest.mark.parametrize(
    ("factory", "returned"),
    [
        ("def create_app():\n    return TokenAuth(_build())\n", "TokenAuth"),
        ("def create_app():\n    app = _build()\n", "NoneType"),
        ("async def create_app():\n    return _build()\n", "coroutine"),
    ],
    ids=["app-wrapped-in-middleware", "returns-none", "async-factory"],
)
@pytest.mark.parametrize("tactic", ["echo", None], ids=["tactic", "no-tactic"])
def test_a_service_entry_that_builds_no_app_or_payload_is_an_error(
    tmp_path: Path, factory: str, returned: str, tactic: str | None
):
    package = _token_auth_package(tmp_path, factory, tactic=tactic)
    message = (
        f"services.api returned {returned}, not a FastAPI or Starlette app, "
        "a tactic, or a data payload"
    )

    with pytest.raises(LaunchError, match=f"^{re.escape(message)}$"):
        load_launch_app(package)


def test_inspect_reports_a_service_wrapped_in_middleware_as_an_error(
    tmp_path: Path, capsys
):
    package = _token_auth_package(
        tmp_path, "def create_app():\n    return TokenAuth(_build())\n", tactic="echo"
    )

    with pytest.raises(SystemExit) as exited:
        main(["inspect", str(package), "--no-keyring"])

    assert exited.value.code == 2
    assert "lllm: error: services.api returned TokenAuth" in capsys.readouterr().err


def test_a_service_without_an_entry_still_serves_its_tactic(tmp_path: Path):
    package = _package(
        tmp_path,
        primary="services.api",
        body="""
[tactics.echo]
entry = "fix2_svc.tactics:Echo"

[services.api]
tactic = "echo"
""",
    )
    _write(package / "fix2_svc" / "tactics.py", _ECHO)

    resolved = load_launch_app(package)

    assert resolved.kind == "service"
    response = TestClient(resolved.app).post("/run", json={"input": "hi"})
    assert response.json()["output"] == {"echo": "hi"}


# 4: served package code can import its own modules when it is called.


def test_a_launched_tactic_can_import_its_package_modules_at_call_time(tmp_path: Path):
    package = _package(
        tmp_path,
        primary="tactics.shout",
        body='\n[tactics.shout]\nentry = "fix2_lazy.tactics:Shout"\n',
        module="fix2_lazy",
    )
    _write(package / "fix2_lazy" / "helpers.py", _HELPERS)
    _write(package / "fix2_lazy" / "tactics.py", _lazy_tactic("fix2_lazy"))

    client = TestClient(load_launch_app(package).app, raise_server_exceptions=False)
    response = client.post("/run", json={"input": "hi"})

    assert response.status_code == 200, response.text
    assert response.json()["output"] == "HI"


def test_a_shell_package_tactic_can_import_its_package_modules_at_call_time(tmp_path: Path):
    package = _package(
        tmp_path,
        primary="tactics.shout",
        body='\n[tactics.shout]\nentry = "fix2_shell.tactics:Shout"\n',
        module="fix2_shell",
        kind="app",
    )
    _write(package / "fix2_shell" / "helpers.py", _HELPERS)
    _write(package / "fix2_shell" / "tactics.py", _lazy_tactic("fix2_shell"))

    app = create_strategy_app(load_strategy(package))
    response = TestClient(app, raise_server_exceptions=False).post(
        "/tactics/shout/run", json={"input": "hi"}
    )

    assert response.status_code == 200, response.text
    assert response.json()["output"] == "HI"


def test_entries_of_one_served_package_share_its_modules(tmp_path: Path):
    package = tmp_path / "pkg"
    _write(package / "fix2_state" / "__init__.py", "")
    _write(package / "fix2_state" / "store.py", "ITEMS = []\n")
    _write(
        package / "fix2_state" / "tactics.py",
        """
from fix2_state.store import ITEMS

def put(value):
    ITEMS.append(value)

def get(value=None):
    from fix2_state import store
    return store.ITEMS
""",
    )

    put = import_entrypoint("fix2_state.tactics:put", base_dir=package, keep_loaded=True)
    get = import_entrypoint("fix2_state.tactics:get", base_dir=package, keep_loaded=True)
    put("first")

    assert get() == ["first"]


def test_a_second_served_package_with_the_same_module_name_gets_its_own_entry(
    tmp_path: Path,
):
    for name in ("alpha", "beta"):
        _write(tmp_path / name / "fix2_same" / "__init__.py", "")
        _write(tmp_path / name / "fix2_same" / "tactics.py", f"NAME = {name!r}\n")

    names = [
        import_entrypoint("fix2_same.tactics:NAME", base_dir=tmp_path / name, keep_loaded=True)
        for name in ("alpha", "beta")
    ]

    assert names == ["alpha", "beta"]


def test_validation_imports_stay_isolated(tmp_path: Path):
    package = _package(
        tmp_path,
        primary="tactics.shout",
        body='\n[tactics.shout]\nentry = "fix2_valid.tactics:Shout"\n',
        module="fix2_valid",
    )
    _write(package / "fix2_valid" / "helpers.py", _HELPERS)
    _write(package / "fix2_valid" / "tactics.py", _lazy_tactic("fix2_valid"))
    path_before = list(sys.path)

    assert validate_package(package).ok
    import_entrypoint("fix2_valid.tactics:Shout", base_dir=package)

    assert sys.path == path_before
    assert not any(name.startswith("fix2_valid") for name in sys.modules)


# 21: one rule tells factories (called once at load) from handlers (called
# once per input), in launch.py and coerce_tactic alike.


def test_a_handler_with_default_arguments_answers_each_input(tmp_path: Path):
    _write(
        tmp_path / "fix2_greet.py",
        """
CALLS = []

def greet(name: str = "world") -> str:
    CALLS.append(name)
    return f"Hello {name}"
""",
    )
    sys.path.insert(0, str(tmp_path))

    app = load_launch_app("fix2_greet:greet").app
    import fix2_greet

    assert fix2_greet.CALLS == []
    response = TestClient(app).post("/run", json={"input": "Ada"})
    assert response.json()["output"] == "Hello Ada"
    assert fix2_greet.CALLS == ["Ada"]


def test_coerce_tactic_does_not_call_handlers_at_load():
    calls = []

    def collect(*args):
        calls.append(args)
        return list(args)

    def greet(name="world"):
        calls.append(name)
        return f"Hello {name}"

    collected = coerce_tactic(collect)
    greeter = coerce_tactic(greet)

    assert calls == []
    assert collected.run("x") == ["x"]
    assert greeter.run("Ada") == "Hello Ada"


def test_factories_are_called_once_at_load(tmp_path: Path):
    _write(
        tmp_path / "fix2_factories.py",
        """
from __future__ import annotations

from fastapi import FastAPI

from lllm.runtimes import as_tactic
from lllm.runtimes.python import CallableTactic

CALLS = []

def build():
    CALLS.append("build")
    return as_tactic(lambda value: f"built:{value}", name="built")

def build_configured(prefix: str = ">") -> CallableTactic:
    CALLS.append("build_configured")
    return as_tactic(lambda value: f"{prefix}{value}", name="configured")

def create_app(debug: bool = False) -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    def health():
        return {"debug": debug}

    return app

class Prefixer:
    def __init__(self, prefix: str = "#"):
        self.prefix = prefix

    def run(self, value):
        return f"{self.prefix}{value}"
""",
    )
    sys.path.insert(0, str(tmp_path))
    import fix2_factories

    assert coerce_tactic(fix2_factories.build).run("x") == "built:x"
    configured = coerce_tactic(fix2_factories.build_configured)
    assert isinstance(configured, CallableTactic)
    assert configured.run("x") == ">x"
    assert fix2_factories.CALLS == ["build", "build_configured"]
    assert coerce_tactic(fix2_factories.Prefixer).run("x") == "#x"
    app = load_launch_app("fix2_factories:create_app").app
    assert TestClient(app).get("/health").json() == {"debug": False}


def test_a_factory_return_annotation_counts_when_a_parameter_annotation_cannot_resolve(
    tmp_path: Path,
):
    _write(
        tmp_path / "fix2_typed.py",
        """
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lllm.protocol import Tactic
from lllm.runtimes import as_tactic

if TYPE_CHECKING:
    from decimal import Context as Settings

CALLS = []

def build_tactic(settings: Settings | None = None) -> Tactic[Any, Any]:
    CALLS.append("build")
    return as_tactic(lambda value: f"built:{value}", name="built")
""",
    )
    sys.path.insert(0, str(tmp_path))
    import fix2_typed

    tactic = coerce_tactic(fix2_typed.build_tactic)
    app = load_launch_app("fix2_typed:build_tactic").app

    assert fix2_typed.CALLS == ["build", "build"]
    assert tactic.run("x") == "built:x"
    response = TestClient(app).post("/run", json={"input": "x"})
    assert response.json()["output"] == "built:x"


def test_a_factory_return_annotation_counts_when_imported_only_for_type_checking(
    tmp_path: Path,
):
    package = _package(
        tmp_path,
        primary="tactics.built",
        module="fix2_tc",
        body='\n[tactics.built]\nentry = "fix2_tc.tactics:build_tactic"\n',
    )
    _write(
        package / "fix2_tc" / "tactics.py",
        """
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lllm.runtimes import as_tactic

if TYPE_CHECKING:
    from fastapi import FastAPI

    from fix2_tc.models import Answer
    from lllm.protocol import Tactic
    from lllm.runtimes import CallableTactic

CALLS = []

def build_tactic(prefix: str = "built:") -> Tactic[Any, Any]:
    CALLS.append("build_tactic")
    return as_tactic(lambda value: f"{prefix}{value}", name="built")

def build_configured(prefix: str = ">") -> CallableTactic:
    CALLS.append("build_configured")
    return as_tactic(lambda value: f"{prefix}{value}", name="configured")

def create_app(debug: bool = False) -> FastAPI:
    import fastapi

    app = fastapi.FastAPI()

    @app.get("/health")
    def health():
        return {"debug": debug}

    return app

def answer(question: str = "") -> Answer:
    CALLS.append(question)
    return {"text": question}
""",
    )
    sys.path.insert(0, str(package))

    response = TestClient(load_launch_app(package).app).post("/run", json={"input": "x"})
    import fix2_tc.tactics as tactics

    assert response.json()["output"] == "built:x"
    assert coerce_tactic(tactics.build_configured).run("x") == ">x"
    coerce_tactic(tactics.answer)
    assert tactics.CALLS == ["build_tactic", "build_configured"]
    app = load_launch_app("fix2_tc.tactics:create_app").app
    assert TestClient(app).get("/health").json() == {"debug": False}


# 22: a Tactic subclass gets name= only when its constructor takes it.


class _V2Summarize(NativeTactic):
    name = "summarize"

    def __init__(self, config=None, runtime=None):
        super().__init__(config, runtime=runtime)

    def call(self, task):
        return task[:5]


class _Named(Tactic[Any, Any]):
    def __init__(self, *, name=None, flag=False):
        super().__init__(name=name)

    def _run(self, input_value, *, context=None):
        return input_value


def test_coerce_tactic_passes_name_only_when_the_constructor_takes_it():
    summarize = coerce_tactic(_V2Summarize, name="brief")
    produced = coerce_tactic(lambda: _V2Summarize, name="brief")

    assert summarize.tactic_name == "summarize"
    assert summarize.run("hello world") == "hello"
    assert isinstance(produced, _V2Summarize)
    assert coerce_tactic(_Named, name="renamed").tactic_name == "renamed"


def test_a_package_tactic_class_without_a_name_parameter_launches(tmp_path: Path):
    package = _package(
        tmp_path,
        primary="tactics.upper",
        body='\n[tactics.upper]\nentry = "fix2_v2.tactics:Upper"\n',
        module="fix2_v2",
    )
    _write(
        package / "fix2_v2" / "tactics.py",
        """
from typing import Any

from lllm.protocol import Tactic


class Upper(Tactic[Any, Any]):
    def __init__(self, config=None):
        super().__init__()
        self.config = config

    def _run(self, input_value, *, context=None):
        return str(input_value).upper()
""",
    )

    app = load_launch_app(package).app
    response = TestClient(app).post("/run", json={"input": "hi"})

    assert response.json()["output"] == "HI"


# 23: `lllm setup` works for a package with nothing to launch.


def test_setup_configures_the_keys_of_a_package_with_nothing_to_launch(
    tmp_path: Path, monkeypatch, capsys
):
    package = tmp_path / "lib"
    _write(
        package / "lllm.toml",
        """
[package]
org = "demo"
name = "lib"
version = "0.1.0"
kind = "library"

[requirements.api_keys]
OPENAI_API_KEY = "Used by the shared proxies."
""",
    )
    env_file = tmp_path / ".env.local"
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    requirements = load_launch_api_key_requirements(package)
    exit_code = main(
        [
            "setup",
            str(package),
            "--credentials",
            "env",
            "--env-file",
            str(env_file),
            "--set",
            "OPENAI_API_KEY=sk-test-secret",
        ]
    )

    assert [requirement.name for requirement in requirements] == ["OPENAI_API_KEY"]
    assert exit_code == 0
    assert "Configured 1 API key" in capsys.readouterr().out
    assert 'OPENAI_API_KEY="sk-test-secret"' in env_file.read_text(encoding="utf-8")
    with pytest.raises(LaunchError, match="does not declare a launchable resource"):
        load_launch_app(package)


# 30, 31: a primary that cannot be launched is a clear LaunchError naming the
# resource and what can be served.


@pytest.mark.parametrize("resource", [None, "package.name", "requirements.api_keys"])
def test_a_config_primary_is_a_clear_launch_error(tmp_path: Path, resource: str | None):
    package = tmp_path / "cfg"
    _write(
        package / "lllm.toml",
        """
[package]
org = "demo"
name = "cfg"
version = "0.1.0"
primary = "config.default"

[config.defaults]
mode = "fast"
""",
    )
    assert validate_package(package).ok
    selected = resource or "config.default"

    with pytest.raises(LaunchError) as raised:
        load_launch_app(package, resource=resource)

    assert f"resource {selected} cannot be launched" in str(raised.value)
    assert "declares no services, tactics, channels, or runs" in str(raised.value)


def test_a_metadata_resource_error_lists_what_can_be_launched(tmp_path: Path, capsys):
    package = _package(
        tmp_path,
        primary="config.default",
        kind="mixed",
        body="""
[config.defaults]
mode = "fast"

[tactics.echo]
entry = "fix2_cfg.tactics:Echo"
""",
        module="fix2_cfg",
    )
    _write(package / "fix2_cfg" / "tactics.py", _ECHO)

    with pytest.raises(SystemExit) as exited:
        main(["inspect", str(package), "--no-keyring"])

    assert exited.value.code == 2
    err = capsys.readouterr().err
    assert "resource config.default cannot be launched" in err
    assert "launchable resources: tactics.echo" in err
    assert load_launch_app(package, resource="tactics.echo").kind == "tactic"


def test_a_snapshot_primary_names_the_channels_that_can_be_launched(tmp_path: Path):
    package = tmp_path / "events"
    _write(
        package / "lllm.toml",
        """
[package]
org = "demo"
name = "events"
version = "0.1.0"
kind = "channel"
primary = "snapshots.latest"

[channels.events]
form = "log"

[snapshots.latest]
channel = "events"
""",
    )
    assert validate_package(package).ok

    with pytest.raises(LaunchError) as raised:
        load_launch_app(package, store=tmp_path / "store")

    assert str(raised.value) == (
        "resource snapshots.latest cannot be launched; launchable resources: channels.events"
    )
    resolved = load_launch_app(package, resource="channels.events", store=tmp_path / "store")
    assert resolved.kind == "channel"


_ECHO = """
class Echo:
    def run(self, input_value, *, context=None):
        return {"echo": input_value}
"""

_TOKEN_AUTH_APP = """
from fastapi import FastAPI


class TokenAuth:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if dict(scope.get("headers") or []).get(b"x-token") != b"secret":
            await send({"type": "http.response.start", "status": 401, "headers": []})
            await send({"type": "http.response.body", "body": b""})
            return
        await self.app(scope, receive, send)


def _build():
    app = FastAPI()

    @app.post("/run")
    def run(body: dict):
        return {"output": body}

    return app

"""

_HELPERS = "def shout(value):\n    return str(value).upper()\n"


def _lazy_tactic(module: str) -> str:
    return f"""
class Shout:
    def run(self, input_value, *, context=None):
        from {module}.helpers import shout

        return shout(input_value)
"""


def _token_auth_package(tmp_path: Path, factory: str, *, tactic: str | None) -> Path:
    tactic_line = f'tactic = "{tactic}"' if tactic else ""
    package = _package(
        tmp_path,
        primary="services.api",
        body=f"""
[tactics.echo]
entry = "fix2_svc.tactics:Echo"

[services.api]
entry = "fix2_svc.app:create_app"
{tactic_line}
""",
    )
    _write(package / "fix2_svc" / "tactics.py", _ECHO)
    _write(package / "fix2_svc" / "app.py", _TOKEN_AUTH_APP + factory)
    return package


def _package(
    tmp_path: Path,
    *,
    body: str,
    primary: str,
    module: str = "fix2_svc",
    kind: str = "tactic",
) -> Path:
    package = tmp_path / "fix2-package"
    _write(package / module / "__init__.py", "")
    _write(
        package / "lllm.toml",
        f"""
[package]
org = "demo"
name = "fix2-package"
version = "0.1.0"
kind = "{kind}"
primary = "{primary}"
description = "Serving fixture."
{body}
""",
    )
    return package


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.lstrip(), encoding="utf-8")
