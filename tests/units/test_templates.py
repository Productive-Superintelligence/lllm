import contextlib
import io
import tempfile
import textwrap
import unittest
from pathlib import Path


BUILTIN_TEMPLATES = {
    "minimal": {
        "config": "default",
        "tactic": "assistant",
        "agents": {"assistant"},
        "resources": [
            "prompts:assistant/system",
            "configs:default",
            "tactics:assistant",
        ],
    },
    "pipeline": {
        "config": "default",
        "tactic": "pipeline",
        "agents": {"planner", "writer", "reviewer"},
        "resources": [
            "prompts:pipeline/planner/system",
            "prompts:pipeline/tasks/plan",
            "configs:default",
            "tactics:pipeline",
        ],
    },
    "service": {
        "config": "default",
        "tactic": "service",
        "agents": {"assistant"},
        "resources": [
            "prompts:service/system",
            "configs:default",
            "tactics:service",
        ],
    },
    "proxy": {
        "config": "default",
        "tactic": "proxy_analyst",
        "agents": {"analyst"},
        "resources": [
            "prompts:analyst/system",
            "proxies:sample",
            "configs:default",
            "tactics:proxy_analyst",
        ],
    },
    "research": {
        "config": "default",
        "tactic": "research",
        "agents": {"researcher", "synthesizer"},
        "resources": [
            "prompts:research/researcher/system",
            "data:topics.yaml",
            "configs:default",
            "tactics:research",
        ],
    },
}


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content), encoding="utf-8")


class TestTemplateResolution(unittest.TestCase):
    def test_builtin_template_resolution(self):
        from lllm.core.templates import list_builtin_template_names, resolve_template

        self.assertEqual(sorted(BUILTIN_TEMPLATES), list_builtin_template_names())

        for template_name in BUILTIN_TEMPLATES:
            with self.subTest(template=template_name):
                source = resolve_template(template_name)

                self.assertEqual(source.kind, "builtin")
                self.assertEqual(source.spec.name, template_name)
                self.assertEqual(source.spec.entry, "template")

    def test_local_template_resolution(self):
        from lllm.core.templates import resolve_template

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "custom"
            _write(
                root / "lllm-template.toml",
                """
                [template]
                name = "custom"
                entry = "template"
                """,
            )
            _write(root / "template" / "README.md", "# Custom")

            source = resolve_template(str(root))

            self.assertEqual(source.kind, "local")
            self.assertEqual(source.spec.name, "custom")

    def test_unknown_template_raises(self):
        from lllm.core.templates import resolve_template

        with self.assertRaises(FileNotFoundError):
            resolve_template("definitely_missing_template")


class TestTemplateRendering(unittest.TestCase):
    def test_manifest_required_variables_are_validated(self):
        from lllm.core.templates import render_template, resolve_template

        source = resolve_template("minimal")
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "project_name"):
                render_template(source, Path(tmp) / "app", {})

    def test_render_replaces_placeholders_and_dotfiles(self):
        from lllm.core.templates import render_template, resolve_template

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            root = tmp_path / "custom"
            _write(
                root / "lllm-template.toml",
                """
                [template]
                name = "custom"
                entry = "template"

                [variables]
                project_name = { required = true }
                """,
            )
            _write(root / "template" / "_gitignore", "{{project_name}}\n")
            _write(root / "template" / "{{project_name}}" / "README.md", "# {{project_name}}\n")

            source = resolve_template(str(root))
            destination = render_template(
                source,
                tmp_path / "rendered",
                {"project_name": "demo-app"},
            )

            self.assertEqual((destination / ".gitignore").read_text(encoding="utf-8"), "demo-app\n")
            self.assertTrue((destination / "demo-app" / "README.md").is_file())

    def test_render_fails_when_destination_exists(self):
        from lllm.core.templates import render_template, resolve_template

        source = resolve_template("minimal")
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "app"
            destination.mkdir()
            with self.assertRaises(FileExistsError):
                render_template(
                    source,
                    destination,
                    {
                        "project_name": "app",
                        "package_name": "app",
                        "pyproject_name": "app",
                        "project_title": "App",
                    },
                )


class TestCreateProject(unittest.TestCase):
    def test_create_project_smoke_all_builtin_templates(self):
        from lllm.cli import create_project
        from lllm import build_tactic, resolve_config
        from lllm.core.config import load_package
        from lllm.core.const import APITypes
        from lllm.core.dialog import Dialog
        from lllm.core.runtime import Runtime
        from lllm.invokers import register_invoker
        from lllm.invokers.base import BaseInvoker

        class DummyInvoker(BaseInvoker):
            def call(
                self,
                dialog: Dialog,
                model: str,
                model_args=None,
                parser_args=None,
                responder: str = "assistant",
                metadata=None,
                api_type: APITypes = APITypes.COMPLETION,
                stream_handler=None,
            ):
                raise AssertionError("Smoke test should not call the LLM.")

        for template_name, expected in BUILTIN_TEMPLATES.items():
            with self.subTest(template=template_name):
                with tempfile.TemporaryDirectory() as tmp:
                    project = create_project(
                        f"{template_name}-app",
                        template_ref=template_name,
                        cwd=tmp,
                    )
                    package_name = f"{template_name}_app"

                    self.assertTrue((project / "lllm.toml").is_file())
                    self.assertTrue((project / ".gitignore").is_file())
                    self.assertTrue((project / ".env.example").is_file())
                    self.assertIn(
                        f'name = "{package_name}"',
                        (project / "lllm.toml").read_text(encoding="utf-8"),
                    )

                    runtime = Runtime()
                    load_package(project / "lllm.toml", runtime=runtime)
                    for resource in expected["resources"]:
                        section, key = resource.split(":", 1)
                        self.assertTrue(
                            runtime.has(f"{package_name}.{section}:{key}"),
                            f"missing {resource}",
                        )

                    config = resolve_config(
                        f"{package_name}:{expected['config']}",
                        runtime=runtime,
                    )
                    config["invoker"] = "dummy"
                    register_invoker("dummy", lambda cfg: DummyInvoker(cfg), overwrite=True)

                    tactic = build_tactic(config, runtime=runtime)

                    self.assertEqual(tactic.name, expected["tactic"])
                    self.assertEqual(set(tactic.agents), expected["agents"])

    def test_create_project_collision(self):
        from lllm.cli import create_project

        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "demo").mkdir()
            with self.assertRaises(FileExistsError):
                create_project("demo", cwd=tmp)

    def test_name_normalization(self):
        from lllm.core.templates import normalize_package_name, normalize_pyproject_name

        self.assertEqual(normalize_package_name("My Cool-App"), "my_cool_app")
        self.assertEqual(normalize_package_name("123 App"), "app_123_app")
        self.assertEqual(normalize_pyproject_name("My Cool_App"), "my-cool-app")


class TestCreateCli(unittest.TestCase):
    def _run_cli(self, *args):
        from lllm.cli import main

        out = io.StringIO()
        err = io.StringIO()
        code = 0
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                main(list(args))
            except SystemExit as exc:
                code = int(exc.code or 0)
        return out.getvalue(), err.getvalue(), code

    def test_cli_create_project(self):
        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path.cwd()
            try:
                import os

                os.chdir(tmp)
                out, err, code = self._run_cli("create", "demo-app")
            finally:
                os.chdir(cwd)

            self.assertEqual(code, 0, err)
            self.assertIn("Created project", out)
            self.assertTrue(Path(tmp, "demo-app", "lllm.toml").is_file())

    def test_cli_name_flag_is_removed(self):
        _, err, code = self._run_cli("create", "--name", "demo")

        self.assertNotEqual(code, 0)
        self.assertIn("unrecognized arguments", err)
