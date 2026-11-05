from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from lllm.launch import load_launch_api_key_requirements, load_launch_app
from lllm.cli import main


def test_launch_tactic_package_as_fastapi(tmp_path: Path):
    package = _write_package(
        tmp_path,
        primary="tactics.echo",
        body="""
[tactics.echo]
entry = "demo.tactics:Echo"
""",
    )
    _write(package / "demo" / "tactics.py", _echo_tactic_source())

    resolved = load_launch_app(package)
    assert resolved.kind == "tactic"
    client = TestClient(resolved.app)
    response = client.post("/run", json={"input": {"message": "hello"}})

    assert response.status_code == 200
    assert response.json()["output"] == {"echo": {"message": "hello"}}


def test_launch_service_package_falls_back_to_declared_tactic(tmp_path: Path):
    package = _write_package(
        tmp_path,
        primary="services.api",
        body="""
[tactics.echo]
entry = "demo.tactics:Echo"

[services.api]
entry = "demo.services:create_app"
tactic = "echo"
transport = "fastapi"

[services.api.metadata]
port = 8123
""",
    )
    _write(package / "demo" / "tactics.py", _echo_tactic_source())
    _write(package / "demo" / "services.py", "def create_app():\n    return {'fixture': True}\n")

    resolved = load_launch_app(package)
    assert resolved.kind == "service"
    assert resolved.port == 8123
    client = TestClient(resolved.app)
    response = client.post("/run", json={"input": "hi"})

    assert response.status_code == 200
    assert response.json()["output"] == {"echo": "hi"}


def test_launch_channel_package_creates_store_api(tmp_path: Path):
    package = _write_package(
        tmp_path,
        kind="channel",
        primary="channels.events",
        body="""
[channels.events]
schema = "lllm://demo/events/schemas/event"
form = "log"
description = "Input events."
""",
    )

    resolved = load_launch_app(package, store=tmp_path / "store")
    assert resolved.kind == "channel"
    client = TestClient(resolved.app)
    response = client.get("/channels")

    assert response.status_code == 200
    assert response.json()[0]["name"] == "events"


def test_launch_service_payload_as_fastapi_run_endpoint(tmp_path: Path):
    package = _write_package(
        tmp_path,
        kind="app",
        primary="services.api",
        body="""
[services.api]
entry = "demo.services:create_app"
transport = "fastapi"
""",
    )
    _write(package / "demo" / "services.py", "def create_app():\n    return {'service': 'demo'}\n")

    resolved = load_launch_app(package)
    client = TestClient(resolved.app)
    response = client.post("/run", json={"input": {"task": "ping"}})

    assert response.status_code == 200
    assert response.json()["output"] == {"service": "demo"}
    assert response.json()["input"] == {"task": "ping"}


def test_launch_direct_fastapi_entrypoint(tmp_path: Path, monkeypatch):
    module = tmp_path / "direct.py"
    module.write_text(
        """
from fastapi import FastAPI


def create_app():
    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"ok": True}

    return app
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    resolved = load_launch_app("direct:create_app")
    client = TestClient(resolved.app)

    assert client.get("/health").json() == {"ok": True}


def test_inspect_cli_reports_launch_surface(tmp_path: Path, capsys):
    package = _write_package(
        tmp_path,
        primary="tactics.echo",
        body="""
[tactics.echo]
entry = "demo.tactics:Echo"
""",
    )
    _write(package / "demo" / "tactics.py", _echo_tactic_source())

    assert main(["inspect", str(package)]) == 0
    out = capsys.readouterr().out

    assert "demo/launch-demo:tactics.echo" in out
    assert "POST /run" in out


def test_package_api_key_requirements_are_discovered(tmp_path: Path):
    package = _write_package(
        tmp_path,
        primary="services.api",
        body="""
[requirements.api_keys]
OPENAI_API_KEY = "OpenAI model access."

[tactics.echo]
entry = "demo.tactics:Echo"

[tactics.echo.metadata.required_api_keys]
ANTHROPIC_API_KEY = "Claude fallback model access."

[services.api]
tactic = "echo"
transport = "fastapi"
""",
    )
    _write(package / "demo" / "tactics.py", _echo_tactic_source())

    requirements = load_launch_api_key_requirements(package)

    assert {requirement.name for requirement in requirements} == {
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
    }


def test_launch_blocks_missing_required_api_key_before_import(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    package = _write_package(
        tmp_path,
        primary="tactics.echo",
        body="""
[requirements.api_keys]
OPENAI_API_KEY = "OpenAI model access."

[tactics.echo]
entry = "demo.tactics:Echo"
""",
    )
    _write(package / "demo" / "tactics.py", "raise RuntimeError('imported too early')\n")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(SystemExit) as exc:
        main(["serve", str(package), "--no-keyring"])

    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "OPENAI_API_KEY" in err
    assert "lllm setup" in err
    assert "imported too early" not in err


def test_setup_writes_required_keys_to_local_env_file(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    package = _write_package(
        tmp_path,
        primary="tactics.echo",
        body="""
[requirements.api_keys]
OPENAI_API_KEY = "OpenAI model access."

[tactics.echo]
entry = "demo.tactics:Echo"
""",
    )
    _write(package / "demo" / "tactics.py", _echo_tactic_source())
    env_file = tmp_path / ".env.local"
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    assert (
        main(
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
        == 0
    )
    out = capsys.readouterr().out

    assert "Configured 1 API key" in out
    assert "sk-test-secret" not in out
    assert 'OPENAI_API_KEY="sk-test-secret"' in env_file.read_text(encoding="utf-8")


def test_inspect_reports_api_key_status_from_env_file(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    package = _write_package(
        tmp_path,
        primary="tactics.echo",
        body="""
[requirements.api_keys]
OPENAI_API_KEY = "OpenAI model access."

[tactics.echo]
entry = "demo.tactics:Echo"
""",
    )
    _write(package / "demo" / "tactics.py", _echo_tactic_source())
    env_file = tmp_path / ".env.local"
    env_file.write_text('OPENAI_API_KEY="sk-test-secret"\n', encoding="utf-8")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    assert main(["inspect", str(package), "--env-file", str(env_file), "--json"]) == 0
    payload = capsys.readouterr().out

    assert '"name": "OPENAI_API_KEY"' in payload
    assert '"ready": true' in payload
    assert "sk-test-secret" not in payload


def _write_package(
    tmp_path: Path,
    *,
    body: str,
    primary: str,
    kind: str = "tactic",
) -> Path:
    package = tmp_path / "launch-demo"
    package.mkdir()
    module = package / "demo"
    module.mkdir()
    _write(module / "__init__.py", "")
    _write(
        package / "lllm.toml",
        f"""
[package]
manifest_version = "0.1"
org = "demo"
name = "launch-demo"
version = "0.1.0"
kind = "{kind}"
primary = "{primary}"
description = "Launch fixture."
{body}
""".lstrip(),
    )
    return package


def _write(path: Path, text: str) -> None:
    path.write_text(text.lstrip(), encoding="utf-8")


def _echo_tactic_source() -> str:
    return """
class Echo:
    def run(self, input_value, *, context=None):
        return {"echo": input_value}
"""
