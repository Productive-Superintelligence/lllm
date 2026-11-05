"""Entrypoint import helpers for local LLLM hub package validation."""

from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


def import_entrypoint(entry: str, *, base_dir: Path | None = None) -> Any:
    if (
        not isinstance(entry, str)
        or not entry
        or entry != entry.strip()
    ):
        raise ValueError(f"Entrypoint must have shape module:attribute: {entry}")
    module_name, sep, attr_path = entry.partition(":")
    if (
        not sep
        or not _entrypoint_segments(module_name)
        or not _entrypoint_segments(attr_path)
    ):
        raise ValueError(f"Entrypoint must have shape module:attribute: {entry}")
    root_module = module_name.split(".", 1)[0]
    with _import_path(base_dir, root_module=root_module):
        module = importlib.import_module(module_name)
        value: Any = module
        for part in attr_path.split("."):
            value = getattr(value, part)
        return value


def _entrypoint_segments(value: str) -> bool:
    return all(
        part
        and not any(ch.isspace() for ch in part)
        and not any(ch in part for ch in "/\\:%")
        for part in value.split(".")
    )


@contextmanager
def _import_path(
    base_dir: Path | None,
    *,
    root_module: str | None = None,
) -> Iterator[None]:
    if base_dir is None:
        yield
        return
    value = str(base_dir)
    isolated_modules: dict[str, Any] = {}
    isolate = root_module is not None and (
        (base_dir / root_module).exists() or (base_dir / f"{root_module}.py").exists()
    )
    if isolate:
        for name in list(sys.modules):
            if name == root_module or name.startswith(f"{root_module}."):
                isolated_modules[name] = sys.modules.pop(name)
    sys.path.insert(0, value)
    importlib.invalidate_caches()
    try:
        yield
    finally:
        try:
            sys.path.remove(value)
        except ValueError:
            pass
        if isolate:
            for name in list(sys.modules):
                if name == root_module or name.startswith(f"{root_module}."):
                    sys.modules.pop(name, None)
            sys.modules.update(isolated_modules)
        importlib.invalidate_caches()
