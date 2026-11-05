"""Declared package file validation helpers."""

from __future__ import annotations

from pathlib import Path, PureWindowsPath
from typing import Any

from .models import PackageManifest, ValidationIssue


def validate_declared_file(
    manifest: PackageManifest,
    *,
    path: str,
    resource: str,
    code: str,
    label: str,
) -> list[ValidationIssue]:
    if manifest.base_dir is None:
        return []
    if invalid_portable_file_path(path):
        return [
            _invalid_path_issue(
                path=path,
                resource=resource,
                code=code,
                label=label,
            )
        ]
    if Path(path).is_absolute() or PureWindowsPath(path).is_absolute():
        return [
            ValidationIssue(
                level="error",
                code=code.replace("_missing", "_absolute_path"),
                message=f"{label} file path must be relative to the package: {path}",
                resource=resource,
            )
        ]
    if ":" in path:
        return [
            _invalid_path_issue(
                path=path,
                resource=resource,
                code=code,
                label=label,
            )
        ]
    base_dir = manifest.base_dir.resolve()
    if has_symlink_component(manifest.base_dir, Path(path)):
        return [
            ValidationIssue(
                level="error",
                code=code.replace("_missing", "_symlink"),
                message=(
                    f"{label} file path must resolve through regular package "
                    f"files, not symlinks: {path}"
                ),
                resource=resource,
            )
        ]
    target = (base_dir / path).resolve()
    if not is_relative_to(target, base_dir):
        return [
            ValidationIssue(
                level="error",
                code=code.replace("_missing", "_outside_package"),
                message=f"{label} file must stay inside the package: {path}",
                resource=resource,
            )
        ]
    if target.is_file():
        return []
    return [
        ValidationIssue(
            level="error",
            code=code,
            message=f"{label} file does not exist: {path}",
            resource=resource,
        )
    ]


def has_symlink_component(base_dir: Path, path: Path) -> bool:
    current = base_dir
    for part in path.parts:
        current = current / part
        if current.is_symlink():
            return True
    return False


def invalid_portable_file_path(path: Any) -> bool:
    if not isinstance(path, str) or not path or path != path.strip():
        return True
    return (
        path.startswith("//")
        or "%" in path
        or "\\" in path
        or "://" in path
        or any(ch.isspace() for ch in path)
    )


def is_relative_to(path: Path, base_dir: Path) -> bool:
    try:
        path.relative_to(base_dir)
    except ValueError:
        return False
    return True


def _invalid_path_issue(
    *,
    path: str,
    resource: str,
    code: str,
    label: str,
) -> ValidationIssue:
    return ValidationIssue(
        level="error",
        code=code.replace("_missing", "_invalid"),
        message=(
            f"{label} file path must be a portable relative path "
            f"without whitespace, percent escapes, URL syntax, "
            f"network-path prefixes, backslashes, or colon separators: {path}"
        ),
        resource=resource,
    )
