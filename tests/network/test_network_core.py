import ast
import os
import subprocess
import sys
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CORE_ROOT = ROOT / "lllm" / "network" / "core"
FORBIDDEN_CORE_IMPORT_PREFIXES = (
    "fastapi",
    "httpx",
    "lllm.hub",
    "lllm.runtimes",
    "lllm.services",
    "lllm.shell",
    "lllm.network.cli",
    "lllm.network.client",
    "lllm.network.resources",
    "lllm.network.server",
    "lllm.network.stores",
)
ALLOWED_CORE_IMPORTS = {
    "__future__",
    "collections.abc",
    "copy",
    "lllm.protocol._validation",
    "pydantic",
    "lllm.network.core.errors",
    "lllm.network.core.models",
    "time",
    "typing",
    "uuid",
}


def test_core_layer_has_no_store_service_or_cross_layer_imports():
    leaks: list[str] = []
    for path in sorted(CORE_ROOT.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for module in _source_imports(tree):
            if module.startswith(FORBIDDEN_CORE_IMPORT_PREFIXES):
                leaks.append(f"{path.relative_to(ROOT)} imports {module}")

    assert leaks == []


def test_core_layer_imports_only_backend_agnostic_dependencies():
    imports: dict[str, list[str]] = {}
    for path in sorted(CORE_ROOT.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for module in _source_imports(tree):
            imports.setdefault(module, []).append(str(path.relative_to(ROOT)))

    unexpected = {
        module: paths
        for module, paths in imports.items()
        if module not in ALLOWED_CORE_IMPORTS
    }

    assert unexpected == {}


def test_top_level_import_does_not_require_optional_service_dependencies(tmp_path):
    _assert_import_while_blocking(
        tmp_path,
        "lllm.network",
        ("fastapi", "httpx", "uvicorn"),
    )


def _source_imports(tree: ast.AST) -> list[str]:
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = _resolve_import_from(node)
            if module:
                modules.append(module)
    return modules


def _resolve_import_from(node: ast.ImportFrom) -> str:
    if node.level == 0:
        return node.module or ""
    if node.level == 1:
        return f"lllm.network.core.{node.module}" if node.module else "lllm.network.core"
    if node.level == 2:
        return f"lllm.network.{node.module}" if node.module else "lllm.network"
    if node.level == 3:
        return f"lllm.{node.module}" if node.module else "lllm"
    return f"<outside-network>.{node.module}" if node.module else "<outside-network>"


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
