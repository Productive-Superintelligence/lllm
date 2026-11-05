from __future__ import annotations

import builtins
from pathlib import Path

from fastapi.testclient import TestClient

from lllm.shell import Strategy, create_strategy_app, load_strategy
from lllm.shell.cli import main


def test_strategy_app_runs_and_invokes_resources():
    strategy = Strategy("analyst-system", description="Fixture system.")
    strategy.channel(
        "events",
        ref="lllm://society/source-channels/channels/finance_ticks",
    )
    strategy.tactic(
        "finance_baseline",
        ref="lllm://society/analyst-tactics/tactics/finance_baseline",
        handler=lambda input_value, context=None: {
            "handled": input_value,
            "context": context or {},
        },
    )

    @strategy.runner
    def run(input_value, *, context=None):
        return {
            "input": input_value,
            "resource_count": len(strategy.resources),
            "context": context or {},
        }

    client = TestClient(create_strategy_app(strategy))

    assert client.get("/health").json() == {
        "ok": True,
        "strategy": "analyst-system",
    }
    assert client.get("/resources").json()[0]["kind"] == "channel"
    run_response = client.post(
        "/run",
        json={"input": {"task": "summarize"}, "context": {"request": "demo"}},
    )
    invoke_response = client.post(
        "/resources/finance_baseline/invoke",
        json={"input": {"records": []}},
    )

    assert run_response.status_code == 200
    assert run_response.json()["output"]["resource_count"] == 2
    assert invoke_response.status_code == 200
    assert invoke_response.json()["output"] == {
        "handled": {"records": []},
        "context": {},
    }


def test_strategy_loader_from_directory(tmp_path: Path):
    strategy_file = tmp_path / "strategy.py"
    strategy_file.write_text(
        """
from lllm.shell import Strategy


def build_strategy():
    strategy = Strategy("loaded")
    strategy.service("api", ref="lllm://demo/app/services/api")
    return strategy
""".lstrip(),
        encoding="utf-8",
    )

    strategy = load_strategy(tmp_path)

    assert strategy.name == "loaded"
    assert strategy.resources[0].kind == "service"


def test_package_directory_loads_manifest_resources_and_tactic(tmp_path: Path):
    package = _write_demo_package(tmp_path / "analysis-pack")

    strategy = load_strategy(package)
    resources = {
        (resource.kind, resource.name): resource
        for resource in strategy.resources
    }
    client = TestClient(create_strategy_app(strategy))

    assert strategy.name == "analysis-pack"
    assert strategy.metadata["package"] == "demo/analysis-pack"
    assert strategy.metadata["card"]["summary"] == "Demo analyst package."
    assert ("package", "analysis-pack") in resources
    assert ("config", "default") in resources
    assert ("schema", "record") in resources
    assert ("tactic", "echo") in resources
    assert ("channel", "events") in resources
    assert ("snapshot", "latest") in resources
    assert ("service", "api") in resources
    assert ("run", "local") in resources
    assert ("doc", "readme") in resources
    assert ("example", "first") in resources
    assert ("asset", "logo") in resources

    response = client.post(
        "/tactics/echo/run",
        json={
            "input": {"text": "hello"},
            "context": {"metadata": {"request": "pkg-test"}},
        },
    )

    assert response.status_code == 200
    assert response.json()["output"] == {
        "echo": {"text": "hello"},
        "context_type": "CallContext",
        "metadata": {"request": "pkg-test"},
    }


def test_package_channels_append_and_query_events(tmp_path: Path):
    package = _write_demo_package(tmp_path / "analysis-pack")
    client = TestClient(create_strategy_app(load_strategy(package)))

    append_response = client.post(
        "/channels/events/events",
        json={
            "input": {
                "payload": {"text": "first"},
                "kind": "record",
                "source": "pytest",
            }
        },
    )
    query_response = client.get("/channels/events/events?limit=10")

    assert append_response.status_code == 200
    assert append_response.json()["output"]["channel"] == "events"
    assert append_response.json()["output"]["payload"] == {"text": "first"}
    assert query_response.status_code == 200
    assert query_response.json()["output"][0]["payload"] == {"text": "first"}
    assert query_response.json()["output"][0]["kind"] == "record"


def test_strategy_can_compose_prefixed_package_resources(tmp_path: Path):
    package = _write_demo_package(tmp_path / "analysis-pack")
    strategy = Strategy("composed").use_package(
        package,
        prefix="pkg",
        store=tmp_path / "store",
    )
    client = TestClient(create_strategy_app(strategy))

    assert any(resource.name == "pkg.echo" for resource in strategy.resources)
    assert any(resource.name == "pkg.events" for resource in strategy.resources)

    response = client.post(
        "/tactics/pkg.echo/run",
        json={"input": {"text": "prefixed"}},
    )

    assert response.status_code == 200
    assert response.json()["output"]["echo"] == {"text": "prefixed"}


def test_tactic_only_package_does_not_import_the_network_module(
    tmp_path: Path,
    monkeypatch,
):
    package = _write_tactic_only_package(tmp_path / "standalone-pack")
    real_import = builtins.__import__

    def blocked_optional_import(name, globals=None, locals=None, fromlist=(), level=0):
        resolved = name
        if level and globals and globals.get("__package__"):
            base = globals["__package__"].rsplit(".", level - 1)[0] if level > 1 else globals["__package__"]
            resolved = f"{base}.{name}" if name else base
        if resolved == "lllm.network" or resolved.startswith("lllm.network."):
            raise ImportError(f"blocked network module: {resolved}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_optional_import)

    strategy = load_strategy(package)
    client = TestClient(create_strategy_app(strategy))
    response = client.post(
        "/tactics/echo/run",
        json={
            "input": {"text": "hello"},
            "context": {"metadata": {"request": "standalone"}},
        },
    )

    assert response.status_code == 200
    assert response.json()["output"] == {
        "echo": {"text": "hello"},
        "context_type": "CallContext",
        "metadata": {"request": "standalone"},
    }


def test_cli_inspect_prints_strategy(tmp_path: Path, capsys):
    strategy_file = tmp_path / "strategy.py"
    strategy_file.write_text(
        """
from lllm.shell import Strategy

strategy = Strategy("cli-demo")
strategy.tactic("review", ref="lllm://demo/pkg/tactics/review")
""".lstrip(),
        encoding="utf-8",
    )

    assert main(["inspect", str(strategy_file)]) == 0
    out = capsys.readouterr().out

    assert "cli-demo" in out
    assert "tactic review lllm://demo/pkg/tactics/review" in out


def test_cli_inspect_prints_package_resources(tmp_path: Path, capsys):
    package = _write_demo_package(tmp_path / "analysis-pack")

    assert main(["inspect", str(package)]) == 0
    out = capsys.readouterr().out

    assert "analysis-pack" in out
    assert "tactic echo lllm://demo/analysis-pack/tactics/echo" in out
    assert "doc readme lllm://demo/analysis-pack/docs/readme" in out


def test_default_runner_exposes_strategy_pack():
    strategy = Strategy("empty")
    strategy.package("analysts", ref="lllm://society/analyst-tactics")

    client = TestClient(create_strategy_app(strategy))
    response = client.post("/run", json={"input": {"ok": True}})

    assert response.status_code == 200
    assert response.json()["output"]["strategy"] == "empty"
    assert response.json()["output"]["resources"][0]["kind"] == "package"


def _write_demo_package(root: Path) -> Path:
    package_dir = root / "demo_pkg"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "schemas.py").write_text(
        """
class Record:
    pass
""".lstrip(),
        encoding="utf-8",
    )
    (package_dir / "tactics.py").write_text(
        """
def echo(input_value, *, context=None):
    metadata = getattr(context, "metadata", context or {})
    context_type = type(context).__name__ if context is not None else None
    return {
        "echo": input_value,
        "context_type": context_type,
        "metadata": metadata,
    }
""".lstrip(),
        encoding="utf-8",
    )
    (root / "README.md").write_text("# Demo Package\n", encoding="utf-8")
    (root / "examples").mkdir()
    (root / "examples" / "first.py").write_text("print('demo')\n", encoding="utf-8")
    (root / "assets").mkdir()
    (root / "assets" / "logo.txt").write_text("LLLM shell\n", encoding="utf-8")
    (root / "lllm.toml").write_text(
        """
[package]
manifest_version = "0.1"
org = "demo"
name = "analysis-pack"
version = "0.1.0"
kind = "app"
primary = "runs.local"
description = "Demo package."

[card]
summary = "Demo analyst package."
tags = ["analysis", "demo"]
suggested_commands = ["lllm serve ."]

[config]
description = "Runtime defaults."
schema = { mode = "string" }
defaults = { mode = "demo" }

[schemas.record]
entry = "demo_pkg.schemas:Record"
description = "Record schema."

[tactics.echo]
entry = "demo_pkg.tactics:echo"
runtime = "python"
input = "schemas.record"
output = "schemas.record"
description = "Echo tactic."
examples = [{ input = { text = "hello" } }]

[channels.events]
form = "log"
description = "Event stream."

[snapshots.latest]
channel = "events"
description = "Latest event."

[services.api]
tactic = "echo"
transport = "fastapi"
description = "HTTP service."
subscribes = ["events"]
publishes = ["events"]

[runs.local]
tactics = ["echo"]
channels = ["events"]
services = ["api"]
snapshots = ["latest"]
description = "Local run."

[docs.readme]
path = "README.md"
title = "README"

[examples.first]
path = "examples/first.py"
command = "python examples/first.py"

[assets.logo]
path = "assets/logo.txt"
media_type = "text/plain"
""".lstrip(),
        encoding="utf-8",
    )
    return root


def _write_tactic_only_package(root: Path) -> Path:
    package_dir = root / "demo_pkg"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "tactics.py").write_text(
        """
def echo(input_value, *, context=None):
    metadata = getattr(context, "metadata", context or {})
    context_type = type(context).__name__ if context is not None else None
    return {
        "echo": input_value,
        "context_type": context_type,
        "metadata": metadata,
    }
""".lstrip(),
        encoding="utf-8",
    )
    (root / "lllm.toml").write_text(
        """
[package]
manifest_version = "0.1"
org = "demo"
name = "standalone-pack"
version = "0.1.0"
kind = "app"
description = "Tactic-only demo package."

[tactics.echo]
entry = "demo_pkg.tactics:echo"
runtime = "python"
description = "Echo tactic."

[runs.local]
tactics = ["echo"]
description = "Local run."
""".lstrip(),
        encoding="utf-8",
    )
    return root


def test_mounted_tactics_get_the_full_service_surface():
    from pydantic import BaseModel

    from lllm import Tactic

    class Text(BaseModel):
        text: str

    class Shout(Tactic[Text, Text]):
        name = "shout"
        input_type = Text
        output_type = Text

        def _run(self, input_value, *, context=None):
            return Text(text=input_value.text.upper())

    strategy = Strategy("served").add_tactic(Shout())
    strategy.tactic("plain", handler=lambda input_value: {"seen": input_value})
    client = TestClient(create_strategy_app(strategy))

    assert [r["name"] for r in client.get("/tactics").json()] == ["shout", "plain"]
    assert client.get("/tactics/shout/info").json()["name"] == "shout"
    assert client.post("/tactics/shout/run", json={"input": {"text": "hi"}}).json()[
        "output"
    ] == {"text": "HI"}
    # Streaming errors come back in the services error envelope.
    stream = client.post("/tactics/shout/stream", json={"input": {"text": "go"}})
    assert stream.status_code == 400
    assert stream.json()["detail"]["error"]["type"] == "TacticUnsupportedError"
    assert client.get("/tactics/shout/info").json()["description"] == ""
    assert client.post("/tactics/plain/run", json={"input": 3}).json()["output"] == {
        "seen": 3
    }


def test_package_channels_are_served_by_the_network_store_api(tmp_path: Path):
    package = _write_demo_package(tmp_path / "analysis-pack")
    strategy = load_strategy(package)
    client = TestClient(create_strategy_app(strategy))

    client.post(
        "/channels/events/events",
        json={"input": {"payload": {"text": "via shell"}, "kind": "record"}},
    )

    assert strategy.store is not None
    assert [c["name"] for c in client.get("/network/channels").json()] == ["events"]
    events = client.get("/network/events", params={"channel": "events"}).json()
    assert [event["payload"] for event in events] == [{"text": "via shell"}]
