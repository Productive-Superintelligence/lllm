import asyncio
import importlib
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import httpx
import pytest

from lllm.cli import load_tactic_entrypoint, main
from lllm.create import create_project

ROOT = Path(__file__).resolve().parents[1]


def test_create_plain_project_builds_runnable_tactic(tmp_path):
    result = create_project("plain", "hello-world", directory=tmp_path)

    assert result.path.name == "hello-world"
    assert result.package_name == "hello_world"
    assert (result.path / "app.py").exists()
    assert (result.path / "client.py").exists()
    assert (result.path / "tests" / "test_tactic.py").exists()
    assert not (result.path / "psi.toml").exists()
    assert _import_generated_client(result.path, result.package_name)
    assert _run_generated_client(result.path, result.package_name) == "HELLO"
    assert _run_generated_tactic(result.path, result.package_name) == "HELLO"


def test_create_pydantic_ai_project_builds_runnable_tactic(tmp_path):
    result = create_project("pydantic-ai", "agent-demo", directory=tmp_path)

    assert _run_generated_tactic(result.path, result.package_name) == "HELLO"


def test_create_native_project_builds_runnable_tactic(tmp_path):
    result = create_project("native", "native-demo", directory=tmp_path)

    assert _run_generated_tactic(result.path, result.package_name) == "HELLO"


@pytest.mark.parametrize("template", ["plain", "pydantic-ai", "native"])
def test_generated_service_app_handles_protocol_endpoints(tmp_path, template):
    result = create_project(template, f"{template}-service", directory=tmp_path)

    responses = _run_generated_service_app(result.path, result.package_name)

    assert responses["health"] == {"ok": True, "tactics": ["echo"]}
    assert responses["info"]["name"] == "echo"
    assert responses["run"] == {
        "output": {"text": "HELLO"},
        "request_id": "req-generated",
        "tactic": "echo",
    }
    assert responses["named_info"]["name"] == "echo"
    assert responses["named_run"]["output"] == {"text": "WORLD"}
    assert responses["named_run"]["tactic"] == "echo"


@pytest.mark.parametrize(
    ("template", "expected_dependency"),
    [
        ("plain", "lllm"),
        ("pydantic-ai", "lllm[pydantic-ai]"),
        ("native", "lllm"),
    ],
)
def test_create_project_metadata_and_docs_match_server_flow(
    tmp_path,
    template,
    expected_dependency,
):
    result = create_project(template, f"{template}-metadata", directory=tmp_path)

    pyproject = tomllib.loads((result.path / "pyproject.toml").read_text())
    assert expected_dependency in pyproject["project"]["dependencies"]
    assert "lllm[server]" in pyproject["project"]["optional-dependencies"]["server"]

    install_command = 'pip install -e ".[dev,server]"'
    serve_command = f"lllm serve {result.package_name}.tactics:build_tactic --port 8000"
    client_command = "python client.py"
    for relative in ("README.md", "docs/tutorial.md"):
        text = (result.path / relative).read_text()
        assert install_command in text
        assert serve_command in text
        assert client_command in text

    client = (result.path / "client.py").read_text()
    assert "RemoteTactic" in client
    assert "http://127.0.0.1:8000/run" in client
    assert "def build_client(" in client
    assert "def call(" in client


def test_generated_project_pytest_suite_runs(tmp_path):
    result = create_project("plain", "suite-demo", directory=tmp_path)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=result.path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_create_refuses_non_empty_project_without_force(tmp_path):
    create_project("plain", "demo", directory=tmp_path)

    with pytest.raises(FileExistsError):
        create_project("plain", "demo", directory=tmp_path)


def test_create_rejects_names_without_letters_or_numbers(tmp_path):
    with pytest.raises(ValueError, match="letter or number"):
        create_project("plain", "___", directory=tmp_path)


def test_create_rejects_percent_bearing_project_names(tmp_path):
    for name in ("demo%2Fname", "demo%20name", "demo%3Aname"):
        with pytest.raises(ValueError, match="percent escapes"):
            create_project("plain", name, directory=tmp_path)


def test_create_rejects_malformed_name_and_directory_values(tmp_path):
    for name in (123, "   "):
        with pytest.raises(ValueError, match="Project name"):
            create_project("plain", name, directory=tmp_path)  # type: ignore[arg-type]

    for directory in (123, "   "):
        with pytest.raises(ValueError, match="directory"):
            create_project("plain", "demo", directory=directory)  # type: ignore[arg-type]


def test_cli_create_project(tmp_path, capsys):
    code = main(["create", "plain", "cli-demo", "--directory", str(tmp_path)])

    assert code == 0
    assert "created plain project" in capsys.readouterr().out
    assert (tmp_path / "cli-demo" / "README.md").exists()


@pytest.mark.parametrize(
    "args",
    [
        ["create", "plain", "___"],
        ["create", "plain", "demo%2Fname"],
        ["create", "plain", "demo", "--directory", "   "],
    ],
)
def test_cli_create_rejects_malformed_project_inputs(args, capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(args)

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Traceback" not in captured.err


@pytest.mark.parametrize(
    "args",
    [
        ["serve", "missing.module:build_tactic", "--host", ""],
        ["serve", "missing.module:build_tactic", "--host", " 127.0.0.1 "],
        ["serve", "missing.module:build_tactic", "--host", "bad host"],
        ["serve", "missing.module:build_tactic", "--host", "http://127.0.0.1"],
        ["serve", "missing.module:build_tactic", "--port", "0"],
        ["serve", "missing.module:build_tactic", "--port", "70000"],
    ],
)
def test_cli_serve_rejects_malformed_bindings_before_import(args, capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(args)

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "serve " in captured.err
    assert "missing.module" not in captured.err


@pytest.mark.parametrize(
    "entrypoint",
    ["demo", " demo:build_tactic ", "demo%2Ftactics:build_tactic"],
)
def test_cli_inspect_rejects_malformed_entrypoint_without_traceback(
    capsys,
    entrypoint,
):
    with pytest.raises(SystemExit) as exc_info:
        main(["inspect", entrypoint])

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Entrypoint must have the form" in captured.err
    assert "Traceback" not in captured.err


@pytest.mark.parametrize(
    "args",
    [
        ["inspect", "missing.module:build_tactic"],
        ["serve", "missing.module:build_tactic"],
    ],
)
def test_cli_reports_missing_entrypoint_modules_without_traceback(args, capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(args)

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "No module named" in captured.err
    assert "Traceback" not in captured.err


def test_cli_reports_missing_entrypoint_attributes_without_traceback(
    tmp_path,
    monkeypatch,
    capsys,
):
    module = tmp_path / "demo_tactics.py"
    module.write_text(
        "def build_tactic():\n"
        "    return lambda value: value\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(SystemExit) as exc_info:
        main(["inspect", "demo_tactics:missing"])

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "missing" in captured.err
    assert "Traceback" not in captured.err


@pytest.mark.parametrize(
    "entrypoint",
    [
        None,
        123,
        "",
        "   ",
        "demo",
        ":build_tactic",
        "demo:",
        "demo..tactics:build_tactic",
        "demo.tactics:build_tactic.",
        "demo.tactics:.build_tactic",
        "demo.tactics:build tactic",
        "demo%2Ftactics:build_tactic",
        "demo.tactics:build%2Ftactic",
        " demo.tactics:build_tactic ",
    ],
)
def test_load_tactic_entrypoint_rejects_malformed_values(entrypoint):
    with pytest.raises(ValueError, match="Entrypoint"):
        load_tactic_entrypoint(entrypoint)  # type: ignore[arg-type]


def _run_generated_tactic(project_path, package_name):
    sys.path.insert(0, str(project_path))
    try:
        module = importlib.import_module(f"{package_name}.tactics")
        result = module.build_tactic().run({"text": "hello"})
        return result.text
    finally:
        sys.path.remove(str(project_path))
        for name in list(sys.modules):
            if name == package_name or name.startswith(f"{package_name}."):
                del sys.modules[name]


def _import_generated_client(project_path, package_name):
    sys.path.insert(0, str(project_path))
    try:
        module = importlib.import_module("client")
        return hasattr(module, "main")
    finally:
        sys.path.remove(str(project_path))
        for name in list(sys.modules):
            if (
                name == "client"
                or name == package_name
                or name.startswith(f"{package_name}.")
            ):
                del sys.modules[name]


def _run_generated_client(project_path, package_name):
    sys.path.insert(0, str(project_path))
    try:
        module = importlib.import_module("client")

        def handler(request):
            assert request.method == "POST"
            assert request.url.path == "/run"
            return httpx.Response(200, json={"output": {"text": "HELLO"}})

        result = module.call(
            "hello",
            url="http://testserver/run",
            transport=httpx.MockTransport(handler),
        )
        return result.text
    finally:
        sys.path.remove(str(project_path))
        for name in list(sys.modules):
            if (
                name == "client"
                or name == package_name
                or name.startswith(f"{package_name}.")
            ):
                del sys.modules[name]


def _run_generated_service_app(project_path, package_name):
    sys.path.insert(0, str(project_path))
    try:
        module = importlib.import_module("app")

        async def run():
            transport = httpx.ASGITransport(app=module.app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                health = await client.get("/health")
                info = await client.get("/info")
                run_response = await client.post(
                    "/run",
                    json={
                        "input": {"text": "hello"},
                        "context": {"request_id": "req-generated"},
                    },
                )
                named_info = await client.get("/tactics/echo/info")
                named_run = await client.post(
                    "/tactics/echo/run",
                    json={"input": {"text": "world"}},
                )

            for response in (health, info, run_response, named_info, named_run):
                response.raise_for_status()
            return {
                "health": health.json(),
                "info": info.json(),
                "run": run_response.json(),
                "named_info": named_info.json(),
                "named_run": named_run.json(),
            }

        return asyncio.run(run())
    finally:
        sys.path.remove(str(project_path))
        for name in list(sys.modules):
            if (
                name == "app"
                or name == package_name
                or name.startswith(f"{package_name}.")
            ):
                del sys.modules[name]
