"""Scaffold tactic/service-first projects."""

from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

TemplateName = Literal["plain", "pydantic-ai", "native"]


@dataclass(frozen=True)
class ScaffoldResult:
    """Files created for a generated project."""

    path: Path
    package_name: str
    template: str
    files: tuple[Path, ...]


def create_project(
    template: TemplateName,
    name: str,
    *,
    directory: str | Path = ".",
    force: bool = False,
) -> ScaffoldResult:
    """Create a new tactic/service project.

    This intentionally does not create `psi.toml`; package initialization
    belongs to PsiHub.
    """

    if template not in {"plain", "pydantic-ai", "native"}:
        raise ValueError("template must be one of: plain, pydantic-ai, native")
    project_slug = _slug(name)
    package_name = _package_name(project_slug)
    root = Path(directory) / project_slug
    if root.exists() and any(root.iterdir()) and not force:
        raise FileExistsError(f"Project directory is not empty: {root}")

    files = _template_files(template, project_slug=project_slug, package_name=package_name)
    created: list[Path] = []
    for relative, content in files.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and not force:
            raise FileExistsError(f"File already exists: {target}")
        target.write_text(content, encoding="utf-8")
        created.append(target)
    return ScaffoldResult(
        path=root,
        package_name=package_name,
        template=template,
        files=tuple(created),
    )


def _template_files(
    template: TemplateName,
    *,
    project_slug: str,
    package_name: str,
) -> dict[str, str]:
    dependency = {
        "plain": "lllm[server]",
        "pydantic-ai": "lllm[pydantic-ai,server]",
        "native": "lllm[server]",
    }[template]
    return {
        "pyproject.toml": _pyproject(project_slug, package_name, dependency),
        "README.md": _readme(project_slug, package_name, template),
        "app.py": _app(package_name),
        f"{package_name}/__init__.py": _init(package_name),
        f"{package_name}/tactics.py": _tactics(template),
        "tests/test_tactic.py": _test(package_name),
        "docs/tutorial.md": _tutorial(project_slug, package_name),
    }


def _pyproject(project_slug: str, package_name: str, dependency: str) -> str:
    return _clean(
        f"""
        [build-system]
        requires = ["setuptools>=69", "wheel"]
        build-backend = "setuptools.build_meta"

        [project]
        name = "{project_slug}"
        version = "0.1.0"
        description = "LLLM tactic service project."
        readme = "README.md"
        requires-python = ">=3.10"
        dependencies = [
            "pydantic>=2.7",
            "{dependency}",
        ]

        [project.optional-dependencies]
        dev = ["pytest>=8.0", "httpx>=0.27"]

        [tool.setuptools.packages.find]
        where = ["."]
        include = ["{package_name}*"]

        [tool.pytest.ini_options]
        testpaths = ["tests"]
        """
    )


def _readme(project_slug: str, package_name: str, template: str) -> str:
    return _clean(
        f"""
        # {project_slug}

        This is an LLLM `{template}` tactic/service project.

        Run tests:

        ```bash
        pip install -e ".[dev]"
        pytest
        ```

        Serve the tactic:

        ```bash
        uvicorn app:app --reload
        ```

        Call it:

        ```bash
        curl -X POST http://127.0.0.1:8000/run \\
          -H 'content-type: application/json' \\
          -d '{{"input":{{"text":"hello"}}}}'
        ```

        The package module is `{package_name}`. Add PsiHub package metadata later
        with `psihub init`.
        """
    )


def _app(package_name: str) -> str:
    return _clean(
        f"""
        from lllm.services import create_tactic_app

        from {package_name}.tactics import build_tactic


        app = create_tactic_app(build_tactic())
        """
    )


def _init(package_name: str) -> str:
    return _clean(
        f"""
        from .tactics import EchoInput, EchoOutput, build_tactic

        __all__ = ["EchoInput", "EchoOutput", "build_tactic"]
        """
    )


def _tactics(template: TemplateName) -> str:
    if template == "pydantic-ai":
        return _clean(
            """
            from pydantic import BaseModel

            from lllm.runtimes import PydanticAITactic


            class EchoInput(BaseModel):
                text: str


            class EchoOutput(BaseModel):
                text: str


            class Result:
                def __init__(self, output):
                    self.output = output


            class DemoAgent:
                name = "demo-agent"
                output_type = str

                def run_sync(self, task, **kwargs):
                    return Result(str(task).upper())


            def build_tactic():
                return PydanticAITactic(
                    DemoAgent(),
                    input_type=EchoInput,
                    output_type=EchoOutput,
                    input_mapper=lambda task: task.text,
                    output_mapper=lambda result: EchoOutput(text=result.output),
                )
            """
        )
    if template == "native":
        return _clean(
            """
            from pydantic import BaseModel

            from lllm.runtimes.native import NativeTacticAdapter


            class EchoInput(BaseModel):
                text: str


            class EchoOutput(BaseModel):
                text: str


            class NativeEcho:
                name = "echo"
                input_model = EchoInput
                output_type = EchoOutput

                def call(self, task):
                    return EchoOutput(text=task.text.upper())


            def build_tactic():
                return NativeTacticAdapter(NativeEcho())
            """
        )
    return _clean(
        """
        from pydantic import BaseModel

        from lllm import Tactic


        class EchoInput(BaseModel):
            text: str


        class EchoOutput(BaseModel):
            text: str


        class EchoTactic(Tactic[EchoInput, EchoOutput]):
            name = "echo"
            input_type = EchoInput
            output_type = EchoOutput

            def _run(self, input_value, *, context=None):
                return EchoOutput(text=input_value.text.upper())


        def build_tactic():
            return EchoTactic()
        """
    )


def _test(package_name: str) -> str:
    return _clean(
        f"""
        from {package_name}.tactics import build_tactic


        def test_tactic_runs():
            result = build_tactic().run({{"text": "hello"}})

            if hasattr(result, "text"):
                assert result.text == "HELLO"
            else:
                assert result == "HELLO"
        """
    )


def _tutorial(project_slug: str, package_name: str) -> str:
    return _clean(
        f"""
        # Tutorial

        This project exposes `{package_name}.tactics:build_tactic` as a FastAPI
        app in `app.py`.

        ```bash
        cd {project_slug}
        pip install -e ".[dev]"
        pytest
        uvicorn app:app --reload
        ```
        """
    )


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip(".-")
    if not slug:
        raise ValueError("Project name must contain at least one letter or number.")
    return slug.lower()


def _package_name(project_slug: str) -> str:
    package = re.sub(r"[^A-Za-z0-9_]+", "_", project_slug).strip("_")
    if not package:
        package = "lllm_tactic"
    if package[0].isdigit():
        package = f"tactic_{package}"
    return package


def _clean(value: str) -> str:
    return textwrap.dedent(value).strip() + "\n"
