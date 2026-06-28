import importlib
import sys

import pytest

from lllm.cli import main
from lllm.create import create_project


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
