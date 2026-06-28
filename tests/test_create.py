import importlib
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from lllm.cli import main
from lllm.create import create_project

ROOT = Path(__file__).resolve().parents[1]


def test_create_plain_project_builds_runnable_tactic(tmp_path):
    result = create_project("plain", "hello-world", directory=tmp_path)

    assert result.path.name == "hello-world"
    assert result.package_name == "hello_world"
    assert (result.path / "app.py").exists()
    assert (result.path / "tests" / "test_tactic.py").exists()
    assert _run_generated_tactic(result.path, result.package_name) == "HELLO"


def test_create_pydantic_ai_project_builds_runnable_tactic(tmp_path):
    result = create_project("pydantic-ai", "agent-demo", directory=tmp_path)

    assert _run_generated_tactic(result.path, result.package_name) == "HELLO"


def test_create_native_project_builds_runnable_tactic(tmp_path):
    result = create_project("native", "native-demo", directory=tmp_path)

    assert _run_generated_tactic(result.path, result.package_name) == "HELLO"


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
    for relative in ("README.md", "docs/tutorial.md"):
        text = (result.path / relative).read_text()
        assert install_command in text
        assert serve_command in text


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


def test_cli_create_project(tmp_path, capsys):
    code = main(["create", "plain", "cli-demo", "--directory", str(tmp_path)])

    assert code == 0
    assert "created plain project" in capsys.readouterr().out
    assert (tmp_path / "cli-demo" / "README.md").exists()


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
