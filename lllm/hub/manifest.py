"""Load and initialize `lllm.toml` manifests."""

from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path
from typing import Any

from .models import PackageInfo, PackageManifest


def require_path_value(value: Any, label: str = "path") -> str:
    try:
        text = os.fspath(value)
    except TypeError as exc:
        raise ValueError(f"{label} must be a non-empty path string") from exc
    if not isinstance(text, str) or not text or text != text.strip():
        raise ValueError(f"{label} must be a non-empty path string")
    return text


def manifest_path(path: str | Path) -> Path:
    return _candidate_manifest_path(path).resolve()


def load_manifest(path: str | Path) -> PackageManifest:
    target = _candidate_manifest_path(path)
    if target.is_symlink():
        raise ValueError("lllm.toml must be a regular package file, not a symlink.")
    target = target.resolve()
    data = _load_toml(target)
    manifest = PackageManifest.model_validate(data)
    manifest.base_dir = target.parent
    return manifest


def _candidate_manifest_path(path: str | Path) -> Path:
    value = Path(require_path_value(path, "manifest path")).expanduser()
    if value.is_dir():
        value = value / "lllm.toml"
    return value


def init_package(
    path: str | Path = ".",
    *,
    name: str | None = None,
    org: str = "local",
    kind: str = "mixed",
    force: bool = False,
) -> Path:
    force_value = _bool_value("force", force)
    root = Path(require_path_value(path, "package path")).expanduser()
    package_name = root.resolve().name if name is None else name
    package = PackageInfo(org=org, name=package_name, kind=kind)
    target = root / "lllm.toml"
    if target.exists() and not force_value:
        return target
    root.mkdir(parents=True, exist_ok=True)
    target.write_text(
        textwrap.dedent(
            f"""
            [package]
            manifest_version = "0.1"
            org = {_toml_string(package.org)}
            name = {_toml_string(package.name)}
            version = {_toml_string(package.version)}
            kind = {_toml_string(package.kind)}
            description = ""

            [card]
            summary = ""

            [docs.readme]
            path = "README.md"
            title = "README"
            """
        ).lstrip(),
        encoding="utf-8",
    )
    readme = root / "README.md"
    if not readme.exists():
        readme.write_text(f"# {package.name}\n\nLLLM package.\n", encoding="utf-8")
    return target


def _bool_value(label: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{label} must be a boolean.")
    return value


def _load_toml(path: Path) -> dict[str, Any]:
    try:
        import tomllib
    except ImportError:  # pragma: no cover - Python 3.10 fallback
        import tomli as tomllib  # type: ignore[no-redef]
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _toml_string(value: str) -> str:
    return json.dumps(value)
