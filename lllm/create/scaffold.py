"""Scaffold new LLLM projects.

Two kinds of templates:

- Project templates (``minimal``, ``pipeline``, ``proxy``, ``research``,
  ``service``, or a local folder with ``lllm-template.toml``): a native LLLM
  project with ``lllm.toml``, prompts, YAML agent configs, a tactic, a
  ``main.py``, and an offline smoke test.
- Service scaffolds (``plain``, ``pydantic-ai``, ``native``): a single typed
  tactic served over HTTP.
"""

from __future__ import annotations

import keyword
import os
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

TemplateName = str
SERVICE_SCAFFOLDS = ("plain", "pydantic-ai", "native")
DEFAULT_TEMPLATE = "minimal"
DEFAULT_MODEL = "gpt-4o-mini"


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
    model: str = DEFAULT_MODEL,
) -> ScaffoldResult:
    """Create a new tactic/service project.

    This does not create `lllm.toml`; run ``lllm init`` when the project
    becomes a package.
    """

    force_value = _bool_value("force", force)
    if template not in SERVICE_SCAFFOLDS:
        return _create_from_template(
            template, name, directory=directory, force=force_value, model=model
        )
    project_slug = _slug(name)
    package_name = _package_name(project_slug)
    root = Path(_path_value(directory, "directory")) / project_slug
    if root.exists() and any(root.iterdir()) and not force_value:
        raise FileExistsError(f"Project directory is not empty: {root}")

    files = _template_files(template, project_slug=project_slug, package_name=package_name)
    created: list[Path] = []
    for relative, content in files.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and not force_value:
            raise FileExistsError(f"File already exists: {target}")
        target.write_text(content, encoding="utf-8")
        created.append(target)
    return ScaffoldResult(
        path=root,
        package_name=package_name,
        template=template,
        files=tuple(created),
    )


def _create_from_template(
    template: str,
    name: str,
    *,
    directory: str | Path,
    force: bool,
    model: str,
) -> ScaffoldResult:
    from .templates import (
        list_builtin_template_names,
        normalize_pyproject_name,
        render_template,
        resolve_template,
        title_from_name,
    )

    try:
        source = resolve_template(template)
    except (FileNotFoundError, ValueError) as exc:
        choices = ", ".join([*list_builtin_template_names(), *SERVICE_SCAFFOLDS])
        raise ValueError(
            f"Unknown template {template!r}. Choose one of: {choices}, "
            "or a folder containing lllm-template.toml."
        ) from exc
    project_slug = _slug(name)
    package_name = _package_name(project_slug)
    root = Path(_path_value(directory, "directory")) / project_slug
    if root.exists() and any(root.iterdir()) and not force:
        raise FileExistsError(f"Project directory is not empty: {root}")
    render_template(
        source,
        root,
        {
            "project_name": project_slug,
            "package_name": package_name,
            "pyproject_name": normalize_pyproject_name(project_slug),
            "project_title": title_from_name(project_slug),
            "model_name": model,
        },
        overwrite=force,
    )
    files = tuple(sorted(path for path in root.rglob("*") if path.is_file()))
    return ScaffoldResult(
        path=root,
        package_name=package_name,
        template=source.spec.name,
        files=files,
    )


def _bool_value(label: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{label} must be a boolean.")
    return value


def _template_files(
    template: TemplateName,
    *,
    project_slug: str,
    package_name: str,
) -> dict[str, str]:
    dependency = {
        "plain": "lllm-core",
        "pydantic-ai": "lllm-core[pydantic-ai]",
        "native": "lllm-core[native]",
    }[template]
    return {
        "pyproject.toml": _pyproject(project_slug, package_name, dependency),
        "README.md": _readme(project_slug, package_name, template),
        "app.py": _app(package_name),
        "client.py": _client(package_name),
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
        server = ["lllm-core[server]"]

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
        pip install -e ".[dev,server]"
        pytest
        ```

        Serve the tactic as a FastAPI app:

        ```bash
        uvicorn app:app --reload
        ```

        Or serve the tactic entrypoint through the LLLM CLI:

        ```bash
        lllm serve {package_name}.tactics:build_tactic --port 8000
        ```

        Call it:

        ```bash
        curl -X POST http://127.0.0.1:8000/run \\
          -H 'content-type: application/json' \\
          -d '{{"input":{{"text":"hello"}}}}'
        ```

        Or call it from Python:

        ```bash
        python client.py
        ```

        The package module is `{package_name}`. Run `lllm init` to add an
        `lllm.toml` package manifest when you want to publish it.
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


def _client(package_name: str) -> str:
    return _clean(
        f"""
        from lllm import RemoteTactic

        from {package_name}.tactics import EchoInput, EchoOutput


        DEFAULT_URL = "http://127.0.0.1:8000/run"


        def build_client(url=DEFAULT_URL, *, transport=None):
            return RemoteTactic(
                url,
                name="echo",
                input_type=EchoInput,
                output_type=EchoOutput,
                transport=transport,
            )


        def call(text="hello", *, url=DEFAULT_URL, transport=None):
            return build_client(url, transport=transport).run({{"text": text}})


        def main():
            result = call()
            print(result.text)


        if __name__ == "__main__":
            main()
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
                    name="echo",
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
        pip install -e ".[dev,server]"
        pytest
        uvicorn app:app --reload
        ```

        ```bash
        lllm serve {package_name}.tactics:build_tactic --port 8000
        ```

        In another terminal, call the service with curl:

        ```bash
        curl -X POST http://127.0.0.1:8000/run \\
          -H 'content-type: application/json' \\
          -d '{{"input":{{"text":"hello"}}}}'
        ```

        Or call it from Python:

        ```bash
        python client.py
        ```
        """
    )


def _path_value(value: Any, label: str) -> str:
    try:
        text = os.fspath(value)
    except TypeError as exc:
        raise ValueError(f"{label} must be a non-empty path string") from exc
    if not isinstance(text, str) or not text or text != text.strip():
        raise ValueError(f"{label} must be a non-empty path string")
    return text


def _slug(value: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("Project name must be a non-empty string.")
    if any(ch in value for ch in "/:\\"):
        raise ValueError("Project name must not contain path separators.")
    if "%" in value:
        raise ValueError("Project name must not contain percent escapes.")
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip(".-")
    if not slug or not re.search(r"[A-Za-z0-9]", slug):
        raise ValueError("Project name must contain at least one letter or number.")
    return slug.lower()


def _package_name(project_slug: str) -> str:
    package = re.sub(r"[^A-Za-z0-9_]+", "_", project_slug).strip("_")
    if not package:
        package = "lllm_tactic"
    if package[0].isdigit() or keyword.iskeyword(package):
        package = f"tactic_{package}"
    return package


def _clean(value: str) -> str:
    return textwrap.dedent(value).strip() + "\n"
