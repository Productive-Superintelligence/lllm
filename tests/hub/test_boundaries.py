import ast
import os
import subprocess
import sys
import textwrap
from pathlib import Path

from lllm.protocol._validation import (
    is_public_sensitive_metadata_key,
    is_schema_metadata_key,
    is_sensitive_metadata_key,
)


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = ROOT / "lllm" / "hub"
PACKAGE_SOURCES = sorted(
    path for path in PACKAGE_ROOT.glob("*.py") if path.name != "__init__.py"
)
FORBIDDEN_IMPORT_ROOTS = (
    "subprocess",
)
FORBIDDEN_CALLS = {
    "asyncio.create_subprocess_exec",
    "asyncio.create_subprocess_shell",
    "os.execl",
    "os.execle",
    "os.execlp",
    "os.execlpe",
    "os.execv",
    "os.execve",
    "os.execvp",
    "os.execvpe",
    "os.posix_spawn",
    "os.posix_spawnp",
    "os.spawnl",
    "os.spawnle",
    "os.spawnlp",
    "os.spawnlpe",
    "os.spawnv",
    "os.spawnve",
    "os.spawnvp",
    "os.spawnvpe",
    "os.system",
    "subprocess.Popen",
    "subprocess.call",
    "subprocess.check_call",
    "subprocess.check_output",
    "subprocess.run",
}


def test_hub_package_does_not_launch_dependency_services():
    violations: list[str] = []
    for path in PACKAGE_SOURCES:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        aliases = _import_aliases(tree)
        for imported in aliases.values():
            if _is_forbidden_import(imported):
                violations.append(f"{path.name} imports {imported}")
        for call in _call_names(tree, aliases):
            if call in FORBIDDEN_CALLS:
                violations.append(f"{path.name} calls {call}")
            if call == "uvicorn.run" and path.name != "cli.py":
                violations.append(f"{path.name} calls {call}")

    assert violations == []


def test_only_cli_serve_launches_the_local_hub_api():
    tree = ast.parse((PACKAGE_ROOT / "cli.py").read_text(encoding="utf-8"))
    aliases = _import_aliases(tree)
    uvicorn_calls = [
        call
        for call in _call_names(tree, aliases)
        if call == "uvicorn.run"
    ]

    assert uvicorn_calls == ["uvicorn.run"]


def test_boundary_scanner_rejects_subprocess_from_imports():
    tree = ast.parse("from subprocess import run as run_process")
    aliases = _import_aliases(tree)

    assert _is_forbidden_import(aliases["run_process"])


def test_top_level_import_does_not_require_optional_service_dependencies(tmp_path):
    _assert_import_while_blocking(
        tmp_path,
        "lllm.hub",
        ("fastapi", "httpx", "uvicorn"),
    )


def test_metadata_key_helpers_preserve_public_and_config_policy_split():
    assert is_sensitive_metadata_key("apiKey")
    assert is_sensitive_metadata_key("sessionCookie")
    assert is_sensitive_metadata_key("set-cookie")
    assert not is_sensitive_metadata_key("apiKeyRef")
    assert not is_sensitive_metadata_key("cookieRef")
    assert not is_sensitive_metadata_key("token_count")

    assert is_public_sensitive_metadata_key("token_count")
    assert is_public_sensitive_metadata_key("sessionCookie")
    assert not is_public_sensitive_metadata_key("cookieRef")
    assert is_schema_metadata_key("inputSchema")


def _import_aliases(tree: ast.AST) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                aliases[alias.asname or alias.name.split(".", 1)[0]] = alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return aliases


def _call_names(tree: ast.AST, aliases: dict[str, str]) -> list[str]:
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _qualified_name(node.func)
            if name:
                parts = name.split(".", 1)
                if parts[0] in aliases:
                    resolved = aliases[parts[0]]
                    name = f"{resolved}.{parts[1]}" if len(parts) > 1 else resolved
                names.append(name)
    return names


def _is_forbidden_import(imported: str) -> bool:
    return any(
        imported == root or imported.startswith(f"{root}.")
        for root in FORBIDDEN_IMPORT_ROOTS
    )


def _qualified_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _qualified_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _assert_import_while_blocking(
    tmp_path: Path,
    package: str,
    optional_modules: tuple[str, ...],
) -> None:
    code = textwrap.dedent(
        f"""
        import importlib
        import importlib.abc
        import sys

        blocked = {optional_modules!r}

        class BlockOptional(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".", 1)[0] in blocked:
                    raise ModuleNotFoundError(
                        f"blocked optional dependency: {{fullname}}"
                    )
                return None

        sys.meta_path.insert(0, BlockOptional())
        module = importlib.import_module({package!r})
        print(module.__name__)
        """
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(ROOT)
        if not env.get("PYTHONPATH")
        else f"{ROOT}{os.pathsep}{env['PYTHONPATH']}"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
